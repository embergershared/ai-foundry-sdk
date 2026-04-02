"""
Context Window Probe – Microsoft Foundry
=========================================
Sends progressively larger prompts to a chosen deployment to discover its
effective context-window limit (128 k, 512 k, 1 M tokens, etc.).

Features
--------
* Token counter  – estimates prompt tokens via tiktoken (falls back to a
                   character-based heuristic when tiktoken has no encoding
                   for the target model).
* Binary-search  – efficiently homes in on the exact context ceiling without
                   hammering the API with thousands of requests.
* Single-shot    – optionally send one prompt of a fixed size and report the
                   result without running the full probe.
* CSV / JSON log – every probe call is recorded for later analysis.

Usage
-----
    # Full binary-search probe (discovers the context limit):
    python context_probe.py

    # Single shot – send a 200 000-token prompt and see what happens:
    python context_probe.py --tokens 200000

    # Target a specific deployment and set a custom upper bound:
    python context_probe.py --deployment gpt-5.4 --max-tokens 1100000

Prerequisites
-------------
    pip install azure-ai-projects>=2.0.0 azure-identity python-dotenv tiktoken
    az login
    AZURE_AI_PROJECT_ENDPOINT set in config.env
"""

import os
import sys
import csv
import json
import math
import time
import argparse
import logging
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ── Load config.env ──────────────────────────────────────────────────────────
_env_path = Path(__file__).parent / "config.env"
load_dotenv(dotenv_path=_env_path, override=False)

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.core.exceptions import HttpResponseError

# ── Optional tiktoken import ──────────────────────────────────────────────────
try:
    import tiktoken  # type: ignore

    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# Well-known context-window sizes to test during the binary search.
CONTEXT_MILESTONES = [
    8_192,
    16_384,
    32_768,
    65_536,
    128_000,
    200_000,
    256_000,
    512_000,
    1_000_000,
    1_048_576,
]

# Tokens reserved for the model's reply so we don't exhaust the window.
REPLY_BUFFER = 256

# Text used to pad the prompt.  Repeated as needed to hit the target token count.
_FILLER_SENTENCE = "The quick brown fox jumps over the lazy dog near the riverbank. "

# ─────────────────────────────────────────────────────────────────────────────
# Token counting
# ─────────────────────────────────────────────────────────────────────────────


def _get_encoding(model: str):
    """Return a tiktoken encoding for *model*, falling back to cl100k_base."""
    if not _TIKTOKEN_AVAILABLE:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def count_tokens(text: str, model: str) -> int:
    """
    Return the estimated token count for *text* given *model*.

    Uses tiktoken when available; falls back to len(text) // 4 (a rough
    character-to-token ratio that works well for English prose).
    """
    enc = _get_encoding(model)
    if enc:
        return len(enc.encode(text))
    # Heuristic fallback: ~4 chars per token for English text
    return max(1, len(text) // 4)


def build_prompt(target_tokens: int, model: str) -> str:
    """
    Build a prompt string whose token count is as close as possible to
    *target_tokens* (minus REPLY_BUFFER).
    """
    goal = max(1, target_tokens - REPLY_BUFFER)
    enc = _get_encoding(model)

    if enc:
        # Accurate path: use tiktoken to hit the target precisely
        filler_tokens = enc.encode(_FILLER_SENTENCE)
        repeats = math.ceil(goal / len(filler_tokens))
        tokens = (filler_tokens * repeats)[:goal]
        body = enc.decode(tokens)
    else:
        # Heuristic path: ~4 chars per token
        body = (_FILLER_SENTENCE * math.ceil(goal * 4 / len(_FILLER_SENTENCE)))[
            : goal * 4
        ]

    prefix = (
        f"[CONTEXT PROBE – target {target_tokens:,} tokens]\n"
        "Please respond with exactly: 'OK <token_count>' where <token_count> "
        "is your best estimate of the number of tokens in this message.\n\n"
    )
    return prefix + body


# ─────────────────────────────────────────────────────────────────────────────
# Token calculator (standalone helper – can be called from the REPL)
# ─────────────────────────────────────────────────────────────────────────────


def token_calculator(text: str, model: str = "gpt-4") -> dict:
    """
    Return a breakdown of token statistics for *text*.

    Example
    -------
    >>> from context_probe import token_calculator
    >>> token_calculator("Hello, world!", model="gpt-4.1")
    """
    token_count = count_tokens(text, model)
    char_count = len(text)
    word_count = len(text.split())
    method = (
        "tiktoken"
        if (_TIKTOKEN_AVAILABLE and _get_encoding(model))
        else "heuristic (÷4)"
    )

    result = {
        "model": model,
        "method": method,
        "characters": char_count,
        "words": word_count,
        "tokens": token_count,
        "fits_128k": token_count <= 128_000,
        "fits_512k": token_count <= 512_000,
        "fits_1M": token_count <= 1_000_000,
    }
    return result


def print_token_report(text: str, model: str) -> None:
    """Pretty-print token_calculator results to stdout."""
    r = token_calculator(text, model)
    sep = "─" * 50
    print(f"\n{sep}")
    print(f"  Token Calculator  ({r['method']})")
    print(sep)
    print(f"  Model      : {r['model']}")
    print(f"  Characters : {r['characters']:>12,}")
    print(f"  Words      : {r['words']:>12,}")
    print(f"  Tokens     : {r['tokens']:>12,}")
    print(f"  Fits 128 k : {'✅' if r['fits_128k'] else '❌'}")
    print(f"  Fits 512 k : {'✅' if r['fits_512k'] else '❌'}")
    print(f"  Fits   1 M : {'✅' if r['fits_1M'] else '❌'}")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# API probe
# ─────────────────────────────────────────────────────────────────────────────


class ProbeResult:
    __slots__ = (
        # ── request ──────────────────────────────────────────────────────
        "tokens_requested",  # target token count we aimed for
        "tokens_actual",  # tokens tiktoken counted in the sent prompt
        # ── response – usage (from API) ──────────────────────────────────
        "usage_prompt_tokens",  # input tokens billed by the API
        "usage_completion_tokens",  # output tokens billed by the API
        "usage_total_tokens",  # total billed tokens
        # ── response – metadata ──────────────────────────────────────────
        "finish_reason",  # stop / length / content_filter / error
        "response_id",  # ID returned by the API
        "model_id",  # model name echoed back by the API
        # ── outcome ──────────────────────────────────────────────────────
        "success",
        "status_code",
        "error",
        "latency_s",
        "tokens_per_second",  # usage_prompt_tokens / latency_s
    )

    def __init__(
        self,
        tokens_requested: int,
        tokens_actual: int,
        success: bool,
        status_code: Optional[int],
        error: Optional[str],
        latency_s: float,
        usage_prompt_tokens: Optional[int] = None,
        usage_completion_tokens: Optional[int] = None,
        usage_total_tokens: Optional[int] = None,
        finish_reason: Optional[str] = None,
        response_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        self.tokens_requested = tokens_requested
        self.tokens_actual = tokens_actual
        self.success = success
        self.status_code = status_code
        self.error = error
        self.latency_s = latency_s
        self.usage_prompt_tokens = usage_prompt_tokens
        self.usage_completion_tokens = usage_completion_tokens
        self.usage_total_tokens = usage_total_tokens
        self.finish_reason = finish_reason
        self.response_id = response_id
        self.model_id = model_id
        billed = usage_prompt_tokens or tokens_actual
        self.tokens_per_second = round(billed / latency_s, 1) if latency_s > 0 else None

    def to_dict(self) -> dict:
        return {
            "tokens_requested": self.tokens_requested,
            "tokens_actual": self.tokens_actual,
            "usage_prompt_tokens": self.usage_prompt_tokens,
            "usage_completion_tokens": self.usage_completion_tokens,
            "usage_total_tokens": self.usage_total_tokens,
            "finish_reason": self.finish_reason,
            "response_id": self.response_id,
            "model_id": self.model_id,
            "success": self.success,
            "status_code": self.status_code,
            "error": self.error,
            "latency_s": round(self.latency_s, 3),
            "tokens_per_second": self.tokens_per_second,
        }


def _extract_usage(response) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Pull (prompt_tokens, completion_tokens, total_tokens) from a response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None, None
    pt = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
    ct = getattr(usage, "output_tokens", None) or getattr(
        usage, "completion_tokens", None
    )
    tt = getattr(usage, "total_tokens", None)
    if tt is None and pt is not None and ct is not None:
        tt = pt + ct
    return pt, ct, tt


def send_probe(openai_client, deployment: str, token_target: int) -> ProbeResult:
    """Send one probe request and return a fully-populated ProbeResult."""
    prompt = build_prompt(token_target, deployment)
    actual_tokens = count_tokens(prompt, deployment)

    log.info(
        "Probing %s – target %3s k / actual %3s k tokens …",
        deployment,
        f"{token_target // 1_000}",
        f"{actual_tokens // 1_000}",
    )

    t0 = time.perf_counter()
    try:
        response = openai_client.responses.create(
            model=deployment,
            input=prompt,
            max_output_tokens=REPLY_BUFFER,
        )
        latency = time.perf_counter() - t0

        pt, ct, tt = _extract_usage(response)

        # finish_reason lives on the first output item for Responses API
        finish_reason = None
        output_items = getattr(response, "output", []) or []
        if output_items:
            finish_reason = getattr(output_items[-1], "stop_reason", None) or getattr(
                output_items[-1], "finish_reason", None
            )
        if finish_reason is None:
            finish_reason = getattr(response, "status", None)

        tps = round((pt or actual_tokens) / latency, 1) if latency > 0 else None
        log.info(
            "  ✅  %.1f s  |  prompt %s tk  |  completion %s tk  |  %s tk/s  →  %s",
            latency,
            f"{pt:,}" if pt else "?",
            f"{ct:,}" if ct else "?",
            f"{tps:,.0f}" if tps else "?",
            getattr(response, "output_text", "")[:60],
        )
        return ProbeResult(
            tokens_requested=token_target,
            tokens_actual=actual_tokens,
            success=True,
            status_code=None,
            error=None,
            latency_s=latency,
            usage_prompt_tokens=pt,
            usage_completion_tokens=ct,
            usage_total_tokens=tt,
            finish_reason=finish_reason,
            response_id=getattr(response, "id", None),
            model_id=getattr(response, "model", None),
        )

    except HttpResponseError as exc:
        latency = time.perf_counter() - t0
        msg = exc.message or str(exc)
        log.warning("  ❌  %s  (HTTP %s)  %.1f s", msg[:120], exc.status_code, latency)
        return ProbeResult(
            tokens_requested=token_target,
            tokens_actual=actual_tokens,
            success=False,
            status_code=exc.status_code,
            error=msg,
            latency_s=latency,
        )

    except Exception as exc:
        latency = time.perf_counter() - t0
        log.warning("  ❌  %s  %.1f s", str(exc)[:120], latency)
        return ProbeResult(
            tokens_requested=token_target,
            tokens_actual=actual_tokens,
            success=False,
            status_code=None,
            error=str(exc),
            latency_s=latency,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Binary-search probe
# ─────────────────────────────────────────────────────────────────────────────


def binary_search_context_limit(
    openai_client,
    deployment: str,
    low: int = 1_000,
    high: int = 1_048_576,
    tolerance: int = 4_096,
) -> tuple[int, list[ProbeResult]]:
    """
    Use binary search to find the largest prompt (in tokens) the deployment
    accepts.  Returns (limit_tokens, [ProbeResult, ...]).
    """
    results: list[ProbeResult] = []
    last_success = 0

    log.info("Starting binary search between %d and %d tokens …", low, high)

    while high - low > tolerance:
        mid = (low + high) // 2
        r = send_probe(openai_client, deployment, mid)
        results.append(r)

        if r.success:
            last_success = mid
            low = mid + 1
        else:
            high = mid - 1

        # Polite pause between requests
        time.sleep(0.5)

    log.info(
        "Binary search complete.  Estimated context limit: ~%d tokens", last_success
    )
    return last_success, results


# ─────────────────────────────────────────────────────────────────────────────
# Milestone probe (quick scan)
# ─────────────────────────────────────────────────────────────────────────────


def milestone_probe(
    openai_client,
    deployment: str,
    step: int = 64_000,
    max_tokens: int = 1_048_576,
) -> tuple[int, list[ProbeResult]]:
    """
    Send prompts in increments of *step* tokens (default 64 k), starting at
    *step* and stopping at the first failure or when *max_tokens* is reached.

    Prints a live table row after each call and returns
    (last_successful_token_count, [ProbeResult, ...]).
    """
    results: list[ProbeResult] = []
    last_success = 0

    # ── table header ─────────────────────────────────────────────────────────
    hdr = (
        f"{'Step':>4}  "
        f"{'Target':>9}  "
        f"{'Actual':>9}  "
        f"{'Prompt tk':>10}  "
        f"{'Compl tk':>9}  "
        f"{'Total tk':>9}  "
        f"{'Latency':>8}  "
        f"{'tk/s':>8}  "
        f"{'Finish':>14}  "
        f"Status"
    )
    sep = "─" * len(hdr)
    print(f"\n{sep}")
    print(hdr)
    print(sep)

    step_num = 0
    target = step
    while target <= max_tokens:
        step_num += 1
        r = send_probe(openai_client, deployment, target)
        results.append(r)

        # ── live table row ────────────────────────────────────────────────
        status_icon = "✅" if r.success else "❌"
        pt_str = (
            f"{r.usage_prompt_tokens:>9,}"
            if r.usage_prompt_tokens is not None
            else f"{'?':>9}"
        )
        ct_str = (
            f"{r.usage_completion_tokens:>8,}"
            if r.usage_completion_tokens is not None
            else f"{'?':>8}"
        )
        tot_str = (
            f"{r.usage_total_tokens:>8,}"
            if r.usage_total_tokens is not None
            else f"{'?':>8}"
        )
        tps_str = (
            f"{r.tokens_per_second:>7,.0f}"
            if r.tokens_per_second is not None
            else f"{'?':>7}"
        )
        fr_str = (r.finish_reason or "-")[:14]
        err_str = f"  ← {r.error[:60]}" if not r.success and r.error else ""

        print(
            f"  {step_num:>2}  "
            f"  {target:>8,}  "
            f"  {r.tokens_actual:>8,}  "
            f" {pt_str}  "
            f" {ct_str}  "
            f"  {tot_str}  "
            f"  {r.latency_s:>6.1f} s  "
            f" {tps_str}  "
            f"  {fr_str:<14}  "
            f"{status_icon}{err_str}"
        )

        if r.success:
            last_success = target
        else:
            log.info("First failure at %d k tokens – stopping scan.", target // 1_000)
            break

        target += step
        time.sleep(0.5)

    print(sep + "\n")
    return last_success, results


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def save_results(
    results: list[ProbeResult],
    deployment: str,
    estimated_limit: int,
    output_dir: Path,
) -> None:
    """Write probe results to JSON and CSV files."""
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = f"probe_{deployment}_{ts}"

    payload = {
        "deployment": deployment,
        "probed_at": ts,
        "estimated_context_limit_tokens": estimated_limit,
        "results": [r.to_dict() for r in results],
    }

    json_path = output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("JSON results → %s", json_path)

    csv_path = output_dir / f"{stem}.csv"
    fieldnames = list(ProbeResult(0, 0, False, None, None, 0).to_dict().keys())
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(r.to_dict() for r in results)
    log.info("CSV  results → %s", csv_path)


def print_summary(results: list[ProbeResult], deployment: str, limit: int) -> None:
    sep = "═" * 70
    wins = [r for r in results if r.success]
    fails = [r for r in results if not r.success]

    print(f"\n{sep}")
    print(f"  Context Probe Summary  –  {deployment}")
    print(sep)
    print(f"  Estimated context limit : ~{limit:>12,} tokens")

    if wins:
        max_ok = max(r.tokens_actual for r in wins)
        print(f"  Largest successful call :  {max_ok:>12,} tokens")
        avg_lat = sum(r.latency_s for r in wins) / len(wins)
        print(f"  Avg latency (successes) :  {avg_lat:>11.1f} s")
        tps_vals = [r.tokens_per_second for r in wins if r.tokens_per_second]
        if tps_vals:
            print(
                f"  Avg throughput          :  {sum(tps_vals)/len(tps_vals):>10,.0f} tk/s"
            )
            print(f"  Peak throughput         :  {max(tps_vals):>10,.0f} tk/s")
        pt_vals = [r.usage_prompt_tokens for r in wins if r.usage_prompt_tokens]
        if pt_vals:
            print(f"  Total prompt tokens     :  {sum(pt_vals):>12,}")
        ct_vals = [r.usage_completion_tokens for r in wins if r.usage_completion_tokens]
        if ct_vals:
            print(f"  Total completion tokens :  {sum(ct_vals):>12,}")

    if fails:
        min_fail = min(r.tokens_actual for r in fails)
        print(f"  Smallest failing call   :  {min_fail:>12,} tokens")
        codes = {r.status_code for r in fails if r.status_code}
        if codes:
            print(
                f"  HTTP error codes seen   :  {', '.join(str(c) for c in sorted(codes))}"
            )

    print(f"  Total API calls made    :  {len(results):>12}")
    print(f"  Successful              :  {len(wins):>12}")
    print(f"  Failed                  :  {len(fails):>12}")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Probe a Foundry deployment's context-window limit."
    )
    p.add_argument(
        "--deployment",
        "-d",
        default=os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", ""),
        help="Deployment name to probe  (default: AZURE_AI_MODEL_DEPLOYMENT_NAME)",
    )
    p.add_argument(
        "--mode",
        "-m",
        choices=["binary", "milestone", "single"],
        default="milestone",
        help=(
            "binary   – binary search between --min-tokens and --max-tokens\n"
            "milestone – test a fixed list of well-known sizes (default)\n"
            "single   – send exactly --tokens tokens and exit"
        ),
    )
    p.add_argument(
        "--tokens",
        "-t",
        type=int,
        default=128_000,
        help="Token count for --mode single  (default: 128000)",
    )
    p.add_argument(
        "--min-tokens",
        type=int,
        default=1_000,
        help="Lower bound for binary search  (default: 1000)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=1_048_576,
        help="Upper bound for binary / milestone search  (default: 1 048 576)",
    )
    p.add_argument(
        "--tolerance",
        type=int,
        default=4_096,
        help="Binary-search stops when high−low ≤ tolerance  (default: 4096)",
    )
    p.add_argument(
        "--calc",
        "-c",
        metavar="TEXT",
        help="Run the token calculator on TEXT and exit (no API calls).",
    )
    p.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path(__file__).parent,
        help="Directory for result files  (default: script directory)",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    # ── Token calculator shortcut (no API needed) ─────────────────────────
    if args.calc:
        model = args.deployment or "gpt-4"
        print_token_report(args.calc, model)
        sys.exit(0)

    # ── Validate deployment ───────────────────────────────────────────────
    if not args.deployment:
        log.error(
            "No deployment specified.  Set AZURE_AI_MODEL_DEPLOYMENT_NAME in "
            "config.env or pass --deployment <name>."
        )
        sys.exit(1)

    endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "")
    if not endpoint:
        log.error("AZURE_AI_PROJECT_ENDPOINT is not set.")
        sys.exit(1)

    log.info("Endpoint   : %s", endpoint)
    log.info("Deployment : %s", args.deployment)
    log.info("Mode       : %s", args.mode)

    credential = DefaultAzureCredential()

    with AIProjectClient(endpoint=endpoint, credential=credential) as project_client:
        with project_client.get_openai_client() as openai_client:

            # ── Single-shot mode ──────────────────────────────────────────
            if args.mode == "single":
                prompt = build_prompt(args.tokens, args.deployment)
                print_token_report(prompt, args.deployment)
                r = send_probe(openai_client, args.deployment, args.tokens)
                results = [r]
                estimated_limit = args.tokens if r.success else 0

            # ── Milestone mode (64 k increments) ─────────────────────────
            elif args.mode == "milestone":
                estimated_limit, results = milestone_probe(
                    openai_client,
                    args.deployment,
                    step=64_000,
                    max_tokens=args.max_tokens,
                )

            # ── Binary-search mode ────────────────────────────────────────
            else:
                estimated_limit, results = binary_search_context_limit(
                    openai_client,
                    args.deployment,
                    low=args.min_tokens,
                    high=args.max_tokens,
                    tolerance=args.tolerance,
                )

            print_summary(results, args.deployment, estimated_limit)
            save_results(results, args.deployment, estimated_limit, args.output_dir)

    log.info("Done.")


if __name__ == "__main__":
    main()
