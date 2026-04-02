"""
Get Deployed Models from Microsoft Foundry

This script uses the Azure AI Projects SDK (v2.x) to connect to a Microsoft Foundry
project and gather details on deployed models, including model name, version, publisher,
capabilities, SKU, and connection info.

Reference: https://learn.microsoft.com/en-us/azure/foundry/how-to/develop/sdk-overview

Prerequisites:
    1. Python >= 3.9
    2. Install required packages:
           pip install azure-ai-projects>=2.0.0 azure-identity
    3. Azure CLI authenticated:  az login
    4. Set the following environment variables (or use a .env file):
           AZURE_AI_PROJECT_ENDPOINT  - Your Foundry project endpoint
               Format: https://<resource-name>.services.ai.azure.com/api/projects/<project-name>
    5. Ensure you have at least the "Azure AI User" RBAC role on the Foundry project.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load environment variables from config.env (sits next to this script)
_env_path = Path(__file__).parent / "config.env"
load_dotenv(dotenv_path=_env_path, override=False)

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ModelDeployment
from azure.core.exceptions import HttpResponseError

# ── Module metadata ───────────────────────────────────────────────────────────
__version__ = "1.0.0"
__all__ = [
    "format_deployment_detail",
    "list_all_deployments",
    "list_deployments_by_publisher",
    "get_single_deployment",
    "list_connections",
    "export_to_json",
]

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)
_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_LOG_DATE_FMT = "%H:%M:%S"


def _configure_logging(
    level: str = "INFO",
    log_file: Path | None = None,
) -> None:
    """Configure root logger with a console handler and an optional file handler.

    Parameters
    ----------
    level:
        One of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR`` (case-insensitive).
    log_file:
        When provided, log records are also written to this path.  Parent
        directories are created automatically.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicated output on re-invocation.
    root.handlers.clear()

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_LOG_DATE_FMT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
            log.debug("File logging enabled → %s", log_file)
        except OSError as exc:
            log.warning("Could not open log file %s: %s", log_file, exc)


# Bootstrap with defaults so callers that import this module get sensible output.
_configure_logging()


def get_project_endpoint() -> str:
    """Retrieve the Foundry project endpoint from environment variables."""
    endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
    if not endpoint:
        log.error(
            "AZURE_AI_PROJECT_ENDPOINT is not set in config.env or the environment.\n"
            "Set it to your Foundry project endpoint, e.g.:\n"
            "  https://<resource-name>.services.ai.azure.com/api/projects/<project-name>"
        )
        sys.exit(1)
    return endpoint


def print_separator(char: str = "=", length: int = 80) -> None:
    """Print a visual separator line."""
    print(char * length)


def format_deployment_detail(deployment) -> dict:
    """Extract deployment details into a dictionary for display / export."""
    detail = {
        "name": getattr(deployment, "name", "N/A"),
        "type": getattr(deployment, "type", "N/A"),
    }

    # ModelDeployment instances carry richer metadata
    if isinstance(deployment, ModelDeployment):
        detail.update(
            {
                "model_name": deployment.model_name or "N/A",
                "model_version": deployment.model_version or "N/A",
                "model_publisher": deployment.model_publisher or "N/A",
                "capabilities": deployment.capabilities or {},
                "sku": deployment.sku or "N/A",
                "connection_name": deployment.connection_name or "N/A",
            }
        )

    return detail


def list_all_deployments(project_client: AIProjectClient) -> list[dict]:
    """List every deployment in the Foundry project and return details."""
    deployments: list[dict] = []

    log.info("Enumerating all model deployments …")
    print_separator()

    for deployment in project_client.deployments.list():
        detail = format_deployment_detail(deployment)
        deployments.append(detail)

        print(f"  Deployment Name   : {detail['name']}")
        print(f"  Type              : {detail['type']}")
        if isinstance(deployment, ModelDeployment):
            print(f"  Model Name        : {detail['model_name']}")
            print(f"  Model Version     : {detail['model_version']}")
            print(f"  Model Publisher   : {detail['model_publisher']}")
            print(f"  Capabilities      : {detail['capabilities']}")
            print(f"  SKU               : {detail['sku']}")
            print(f"  Connection Name   : {detail['connection_name']}")
        print_separator("-")

    if not deployments:
        log.warning("No deployments found in this project.")

    log.info("Found %d deployment(s).", len(deployments))
    return deployments


def list_deployments_by_publisher(
    project_client: AIProjectClient, publisher: str
) -> list[dict]:
    """List deployments filtered by model publisher."""
    deployments: list[dict] = []

    log.info("Deployments by publisher '%s':", publisher)
    print_separator()

    for deployment in project_client.deployments.list(model_publisher=publisher):
        detail = format_deployment_detail(deployment)
        deployments.append(detail)
        print(
            f"  {detail['name']}  |  {detail.get('model_name', 'N/A')}  |  {detail.get('model_version', 'N/A')}"
        )

    if not deployments:
        log.warning("No deployments found for publisher '%s'.", publisher)

    print_separator("-")
    return deployments


def get_single_deployment(
    project_client: AIProjectClient, deployment_name: str
) -> dict | None:
    """Retrieve and display a single deployment by name."""
    log.info("Getting deployment '%s' …", deployment_name)
    print_separator()

    try:
        deployment = project_client.deployments.get(deployment_name)
        detail = format_deployment_detail(deployment)

        for key, value in detail.items():
            print(f"  {key:<20}: {value}")

        print_separator("-")
        return detail

    except HttpResponseError as exc:
        log.warning(
            "Could not retrieve deployment '%s' (HTTP %s): %s",
            deployment_name,
            exc.status_code,
            exc.message or str(exc),
        )
        return None

    except Exception as exc:
        log.exception(
            "Unexpected error fetching deployment '%s': %s", deployment_name, exc
        )
        return None


def list_connections(project_client: AIProjectClient) -> None:
    """List all connected resources in the Foundry project."""
    log.info("Enumerating connected resources …")
    print_separator()

    try:
        connections = list(project_client.connections.list())
    except HttpResponseError as exc:
        log.warning(
            "Could not list connections (HTTP %s): %s",
            exc.status_code,
            exc.message or str(exc),
        )
        print_separator("-")
        return
    except Exception as exc:
        log.exception("Unexpected error listing connections: %s", exc)
        print_separator("-")
        return

    for connection in connections:
        print(f"  {connection}")

    if not connections:
        log.warning("No connected resources found.")

    print_separator("-")


def export_to_json(deployments: list[dict], output_path: Path) -> None:
    """Export deployment details to a JSON file.

    The parent directory is created automatically if it does not exist.
    """
    output = {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_deployments": len(deployments),
        "deployments": deployments,
    }
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, indent=2, default=str), encoding="utf-8"
        )
        log.info("Results exported → %s", output_path)
    except OSError as exc:
        log.error("Failed to write JSON results to %s: %s", output_path, exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and export deployed models from a Microsoft Foundry project.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--deployment",
        "-d",
        default=os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", ""),
        help="Look up this specific deployment by name in addition to the full listing\n"
        "(default: AZURE_AI_MODEL_DEPLOYMENT_NAME env var).",
    )
    parser.add_argument(
        "--publisher",
        "-p",
        default="",
        help="Filter the by-publisher breakdown to this publisher only.\n"
        "Leave empty to enumerate every unique publisher discovered.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path(__file__).parent,
        help="Directory for the exported JSON file  (default: script directory).",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        metavar="LEVEL",
        help="Logging verbosity: DEBUG | INFO | WARNING | ERROR  (default: INFO).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Configure logging ─────────────────────────────────────────────────
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_file = args.output_dir / f"get_deployed_models_{ts}.log"
    _configure_logging(level=args.log_level, log_file=log_file)

    # ── Resolve endpoint ──────────────────────────────────────────────────
    endpoint = get_project_endpoint()

    log.info("Endpoint   : %s", endpoint)
    log.debug("Output dir : %s", args.output_dir)

    print_separator()
    print("  Microsoft Foundry – Deployed Model Inspector")
    print(f"  Endpoint: {endpoint}")
    print_separator()

    # ── Authenticate ──────────────────────────────────────────────────────
    try:
        credential = DefaultAzureCredential()
    except Exception as exc:
        log.exception("Failed to create Azure credential: %s", exc)
        sys.exit(1)

    # ── Run inspection ────────────────────────────────────────────────────
    try:
        with AIProjectClient(
            endpoint=endpoint, credential=credential
        ) as project_client:

            # ── 1. List all deployments ──────────────────────────────────────
            all_deployments = list_all_deployments(project_client)

            # ── 2. Summarise unique publishers ───────────────────────────────
            publishers: set[str] = {
                d["model_publisher"]
                for d in all_deployments
                if d.get("model_publisher") and d.get("model_publisher") != "N/A"
            }
            if publishers:
                target_publishers = (
                    {args.publisher} if args.publisher else sorted(publishers)
                )
                log.info("Unique publishers found: %s", ", ".join(sorted(publishers)))
                for pub in target_publishers:
                    list_deployments_by_publisher(project_client, pub)

            # ── 3. Look up a specific deployment (optional) ──────────────────
            if args.deployment:
                get_single_deployment(project_client, args.deployment)

            # ── 4. List connected resources ──────────────────────────────────
            list_connections(project_client)

            # ── 5. Export results to JSON ────────────────────────────────────
            output_path = args.output_dir / "deployed_models.json"
            export_to_json(all_deployments, output_path)

    except KeyboardInterrupt:
        log.warning("Interrupted by user (Ctrl-C).")
        sys.exit(130)

    except HttpResponseError as exc:
        log.error(
            "Azure API error (HTTP %s): %s",
            exc.status_code,
            exc.message or str(exc),
        )
        sys.exit(1)

    except Exception as exc:
        log.exception("Unexpected error: %s", exc)
        sys.exit(1)

    log.info("Done.")


if __name__ == "__main__":
    main()
