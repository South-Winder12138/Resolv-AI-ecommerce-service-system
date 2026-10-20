"""Ordinary tests must deny network before real services or downloads can run."""

import os
import socket
import subprocess
import sys

import pytest


def test_offline_tcp_dns_and_python_child():
    if os.environ.get("MH_HYBRID_SNAPSHOT_SOURCE"):
        pytest.skip("Committed export records its own separate TCP/DNS probes")
    with pytest.raises(RuntimeError, match="network blocked"):
        socket.getaddrinfo("offline-test.invalid", 443)
    with socket.socket(socket.AF_INET) as client, pytest.raises(RuntimeError, match="network blocked"):
        client.connect(("127.0.0.1", 9))
    result = subprocess.run([sys.executable, "-c", "import socket; socket.getaddrinfo('offline-child.invalid',443)"], capture_output=True, text=True, check=False)
    assert result.returncode != 0 and "network blocked" in result.stderr
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["PYTHON_DOTENV_DISABLED"] == "1"
