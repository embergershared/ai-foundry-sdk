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

import os
import sys
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from config.env (sits next to this script)
_env_path = Path(__file__).parent / "config.env"
load_dotenv(dotenv_path=_env_path, override=False)

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ModelDeployment
from azure.core.exceptions import HttpResponseError


def get_project_endpoint() -> str:
    """Retrieve the Foundry project endpoint from environment variables."""
    endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
    if not endpoint:
        print("ERROR: Environment variable AZURE_AI_PROJECT_ENDPOINT is not set.")
        print("       Set it to your Foundry project endpoint, e.g.:")
        print(
            "       https://<resource-name>.services.ai.azure.com/api/projects/<project-name>"
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

    print("\n📋  Enumerating all model deployments …\n")
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
        print("  (no deployments found)")

    return deployments


def list_deployments_by_publisher(
    project_client: AIProjectClient, publisher: str
) -> list[dict]:
    """List deployments filtered by model publisher."""
    deployments: list[dict] = []

    print(f"\n🔍  Deployments by publisher '{publisher}':\n")
    print_separator()

    for deployment in project_client.deployments.list(model_publisher=publisher):
        detail = format_deployment_detail(deployment)
        deployments.append(detail)
        print(
            f"  {detail['name']}  |  {detail.get('model_name', 'N/A')}  |  {detail.get('model_version', 'N/A')}"
        )

    if not deployments:
        print(f"  (no deployments found for publisher '{publisher}')")

    print_separator("-")
    return deployments


def get_single_deployment(
    project_client: AIProjectClient, deployment_name: str
) -> dict | None:
    """Retrieve and display a single deployment by name."""
    print(f"\n🎯  Getting deployment '{deployment_name}':\n")
    print_separator()

    try:
        deployment = project_client.deployments.get(deployment_name)
        detail = format_deployment_detail(deployment)

        for key, value in detail.items():
            print(f"  {key:<20}: {value}")

        print_separator("-")
        return detail

    except HttpResponseError as exc:
        print(f"  ⚠️  Could not retrieve deployment '{deployment_name}': {exc.message}")
        return None


def list_connections(project_client: AIProjectClient) -> None:
    """List all connected resources in the Foundry project."""
    print("\n🔗  Connected resources:\n")
    print_separator()

    for connection in project_client.connections.list():
        print(f"  {connection}")

    print_separator("-")


def export_to_json(deployments: list[dict], filepath: str) -> None:
    """Export deployment details to a JSON file."""
    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_deployments": len(deployments),
        "deployments": deployments,
    }
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"\n💾  Results exported to {filepath}")


def main() -> None:
    endpoint = get_project_endpoint()

    print_separator()
    print("  Microsoft Foundry – Deployed Model Inspector")
    print(f"  Endpoint: {endpoint}")
    print_separator()

    # Authenticate using DefaultAzureCredential (supports az login, managed identity, etc.)
    credential = DefaultAzureCredential()

    with AIProjectClient(endpoint=endpoint, credential=credential) as project_client:

        # ── 1. List all deployments ──────────────────────────────────────────
        all_deployments = list_all_deployments(project_client)

        # ── 2. Summarise unique publishers ───────────────────────────────────
        publishers: set[str] = {
            d["model_publisher"]
            for d in all_deployments
            if d.get("model_publisher") and d.get("model_publisher") != "N/A"
        }
        if publishers:
            print(f"\n📦  Unique publishers found: {', '.join(sorted(publishers))}")
            for pub in sorted(publishers):
                list_deployments_by_publisher(project_client, pub)

        # ── 3. Look up a specific deployment (optional) ──────────────────────
        specific_name = os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME")
        if specific_name:
            get_single_deployment(project_client, specific_name)

        # ── 4. List connected resources ──────────────────────────────────────
        list_connections(project_client)

        # ── 5. Export results to JSON ────────────────────────────────────────
        output_path = os.path.join(
            os.path.dirname(__file__) or ".", "deployed_models.json"
        )
        export_to_json(all_deployments, output_path)

    print("\n✅  Done.\n")


if __name__ == "__main__":
    main()
