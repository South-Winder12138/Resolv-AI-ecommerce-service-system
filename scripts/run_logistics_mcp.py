"""Independent stateless MCP Server, with no direct dataset lookup."""

import argparse
from pathlib import Path

from scripts.logistics_gate_support import (
    owned_profile,
    serve_owned_app,
    validate_service_port,
)

from resolv.services.logistics_api_client import LogisticsAPIClient
from resolv.services.logistics_mcp_server import create_logistics_mcp_app


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--ownership", required=True, type=Path)
    value.add_argument("--port", required=True, type=int)
    return value


def create_app(ownership, port):
    validate_service_port(ownership, "mcp", port)
    profile = owned_profile(ownership)
    return create_logistics_mcp_app(profile, LogisticsAPIClient(profile))


def main():
    args = parser().parse_args()
    app = create_app(args.ownership, args.port)
    serve_owned_app(app, args.ownership, "mcp", args.port)


if __name__ == "__main__":
    main()
