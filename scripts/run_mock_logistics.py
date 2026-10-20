"""Independent loopback-only deterministic logistics API."""

import argparse
from pathlib import Path

from scripts.logistics_gate_support import serve_owned_app, validate_service_port

from resolv.api.mock_logistics import create_mock_logistics_app
from resolv.services.logistics_dataset import load_dataset


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--ownership", required=True, type=Path)
    value.add_argument("--port", required=True, type=int)
    return value


def create_app(ownership, port):
    obj = validate_service_port(ownership, "api", port)
    dataset = load_dataset(Path(obj["source_root"]) / "logistics/datasets/mock-logistics-v1.json")
    return create_mock_logistics_app(dataset, fault=obj["api_fault"])


def main():
    args = parser().parse_args()
    app = create_app(args.ownership, args.port)
    serve_owned_app(app, args.ownership, "api", args.port)


if __name__ == "__main__":
    main()
