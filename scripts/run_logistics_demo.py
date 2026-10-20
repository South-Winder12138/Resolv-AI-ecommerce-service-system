"""Fake-inference local demo. Real models require the bounded acceptance runner."""

import argparse
from pathlib import Path

from scripts.logistics_gate_support import serve_owned_app, validate_service_port

from resolv.services.logistics_demo import create_logistics_demo_app


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--ownership", required=True, type=Path)
    value.add_argument("--port", required=True, type=int)
    return value


def create_app(ownership, port):
    validate_service_port(ownership, "resolv", port)
    return create_logistics_demo_app(ownership)


def main():
    args = parser().parse_args()
    app = create_app(args.ownership, args.port)
    serve_owned_app(app, args.ownership, "resolv", args.port)


if __name__ == "__main__":
    main()
