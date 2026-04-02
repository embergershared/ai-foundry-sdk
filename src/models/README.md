# Foundry Model Lister & Context-Window Probe

A set of Python scripts that connect to a **Microsoft Foundry** project via the
[Azure AI Projects SDK (v2.x)](https://learn.microsoft.com/en-us/azure/foundry/how-to/develop/sdk-overview?pivots=programming-language-python)
to enumerate deployed models and probe their context-window limits.

---

## Repository layout

| File | Description |
|---|---|
| `get_deployed_models.py` | Lists every model deployment in the Foundry project (name, version, publisher, capabilities, SKU) and exports the results to `deployed_models.json`. |
| `context_probe.py` | Sends progressively larger prompts (64 k increments) to a deployment to discover the effective context-window limit. Includes a built-in token calculator. |
| `config.env` | Environment variables consumed by both scripts (not committed with real values). |
| `requirements.txt` | Python dependencies. |
| `.vscode/launch.json` | VS Code / debugpy launch configurations for all scripts and modes. |

---

## Prerequisites

| Requirement | Details |
|---|---|
| Python | ≥ 3.9 |
| Azure CLI | Authenticated (`az login`) |
| RBAC role | At least **Azure AI User** on the Foundry project |
| Foundry endpoint | Format: `https://<resource>.services.ai.azure.com/api/projects/<project>` |

---

## Quick start

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment (edit config.env with your values)
#    AZURE_AI_PROJECT_ENDPOINT = "https://<resource>.services.ai.azure.com/api/projects/<project>"
#    AZURE_AI_MODEL_DEPLOYMENT_NAME = "gpt-5.4"   (optional)

# 4. Authenticate
az login

# 5. List all deployed models
python get_deployed_models.py

# 6. Probe context-window limit (64 k increments)
python context_probe.py --mode milestone

# 7. Or run a single-shot test at a specific token count
python context_probe.py --mode single --tokens 200000
```

---

## `config.env` reference

```dotenv
# Required – Foundry project endpoint
AZURE_AI_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"

# Optional – deployment to look up individually / probe
AZURE_AI_MODEL_DEPLOYMENT_NAME="gpt-5.4"
```

Both scripts load `config.env` automatically via `python-dotenv`.
Shell environment variables take precedence over the file (`override=False`).

---

## `get_deployed_models.py`

Enumerates all model deployments in the Foundry project and writes
`deployed_models.json`.

### What it reports per deployment

- Deployment name & type
- Model name, version, publisher
- Capabilities (chat completion, embeddings, etc.)
- SKU & provisioned capacity
- Connection name

### Example output (excerpt)

| Deployment | Model | Version | Publisher | SKU / Capacity |
|---|---|---|---|---|
| gpt-4.1 | gpt-4.1 | 2025-04-14 | OpenAI | GlobalStandard / 500 |
| gpt-5.4 | gpt-5.4 | 2026-03-05 | OpenAI | GlobalStandard / 500 |
| grok-4 | grok-4 | 1 | xAI | GlobalStandard / 500 |
| DeepSeek-V3.2 | DeepSeek-V3.2 | 1 | DeepSeek | GlobalStandard / 500 |
| Mistral-Large-3 | Mistral-Large-3 | 1 | Mistral AI | GlobalStandard / 500 |
| Kimi-K2.5 | Kimi-K2.5 | 1 | MoonshotAI | GlobalStandard / 50 |
| text-embedding-ada-002 | text-embedding-ada-002 | 2 | OpenAI | GlobalStandard / 250 |

*(18 deployments total — see `deployed_models.json` for the full list)*

---

## `context_probe.py`

### Modes

| Mode | CLI flag | Behaviour |
|---|---|---|
| **Milestone** *(default)* | `--mode milestone` | Sends prompts at 64 k, 128 k, 192 k, … until the model rejects the request. |
| **Binary search** | `--mode binary` | Binary-searches between `--min-tokens` and `--max-tokens` for the exact limit. |
| **Single shot** | `--mode single --tokens N` | Sends one prompt of exactly *N* tokens and reports the result. |

### Token calculator (offline, no API call)

```powershell
python context_probe.py --calc "Your text here" --deployment gpt-5.4
```

Prints character count, word count, estimated token count (via `tiktoken`),
and whether the text fits within 128 k / 512 k / 1 M token windows.

### Tracked metrics per probe call

| Metric | Source |
|---|---|
| `tokens_requested` | Target token count the probe aimed for |
| `tokens_actual` | Tokens counted by tiktoken in the sent prompt |
| `usage_prompt_tokens` | Input tokens reported by the API |
| `usage_completion_tokens` | Output tokens reported by the API |
| `usage_total_tokens` | Total billed tokens from the API |
| `latency_s` | Wall-clock time for the API call |
| `tokens_per_second` | `usage_prompt_tokens / latency_s` |
| `finish_reason` | `completed`, `length`, `content_filter`, etc. |
| `response_id` | API response identifier |
| `model_id` | Model name echoed back by the API |
| `status_code` | HTTP status code (on failure) |
| `error` | Error message (on failure) |

Results are saved as both `.json` and `.csv` after every run.

---

## Probe results — `gpt-5.4` (64 k increments)

Run date: **2026-04-02**

| Step | Requested | Actual tokens | API prompt tk | API compl. tk | Latency (s) | Throughput (tk/s) | Status |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 64,000 | 59,536 | 59,543 | 8 | 15.9 | 3,752 | ✅ |
| 2 | 128,000 | 119,269 | 119,276 | 8 | 3.1 | 38,789 | ✅ |
| 3 | 192,000 | 179,003 | 179,010 | 8 | 4.6 | 38,754 | ✅ |
| 4 | 256,000 | 238,736 | 238,743 | 8 | 5.6 | 42,659 | ✅ |
| 5 | 320,000 | 298,469 | 298,476 | 8 | 6.5 | 45,893 | ✅ |
| 6 | 384,000 | 358,203 | 358,210 | 8 | 46.2 | 7,758 | ✅ |
| 7 | 448,000 | 417,936 | 417,943 | 8 | 49.8 | 8,397 | ✅ |
| 8 | 512,000 | 477,669 | 477,676 | 8 | 9.1 | 52,744 | ✅ |
| 9 | 576,000 | 537,403 | 537,410 | 8 | 13.4 | 39,979 | ✅ |
| 10 | 640,000 | 597,136 | 597,143 | 8 | 12.4 | 48,347 | ✅ |
| 11 | 704,000 | 656,869 | 656,876 | 8 | 14.0 | 47,017 | ✅ |
| 12 | 768,000 | 716,603 | 716,610 | 8 | 14.5 | 49,260 | ✅ |
| 13 | 832,000 | 776,336 | 776,343 | 8 | 16.6 | 46,799 | ✅ |
| 14 | 896,000 | 836,069 | 836,076 | 8 | 18.1 | 46,123 | ✅ |
| 15 | 960,000 | 895,803 | 895,810 | 8 | 19.1 | 47,009 | ✅ |
| 16 | 1,024,000 | 955,538 | — | — | 5.5 | — | ❌ |

> **Estimated context limit: ~960,000 tokens** (between 895,810 and 955,538 API-counted prompt tokens).
>
> Failure error: *"Your input exceeds the context window of this model."*
> (`invalid_request_error` / `context_length_exceeded`)
>
> Note: OpenAI's context window definition includes both input and output tokens.

### Summary statistics (successful calls only)

| Metric | Value |
|---|---|
| Total prompt tokens consumed | 6,367,152 |
| Total completion tokens consumed | 120 |
| Average latency | 16.5 s |
| Average throughput | 36,885 tk/s |
| Peak throughput | 52,744 tk/s |

---

## VS Code debug configurations

Select from the **Run & Debug** panel:

| Configuration | Script | Args |
|---|---|---|
| Debug | `get_deployed_models.py` | — |
| Context Probe – Milestone scan | `context_probe.py` | `--mode milestone` |
| Context Probe – Binary search | `context_probe.py` | `--mode binary --max-tokens 1048576` |
| Context Probe – Single shot (128 k) | `context_probe.py` | `--mode single --tokens 128000` |

All configurations use the `.venv` Python environment and load `config.env` automatically.
