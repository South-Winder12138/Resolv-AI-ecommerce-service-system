"""Opt-in private failure artifacts under the existing refund-owned workspace."""

import json
import os
import re
from pathlib import Path

from scripts import refund_gate_support as s


class OwnedRefundDiagnostics:
    def __init__(self, manifest, report_dir):
        self.manifest = Path(manifest)
        self.owner = s.validate_manifest(self.manifest)["owner"]
        self.target = Path(report_dir).absolute()
        root = (s.ROOT / s.WORKSPACE).resolve()
        if (
            self.target == root
            or self.target.resolve() != self.target
            or not self.target.is_relative_to(root)
            or not self.target.is_dir()
        ):
            raise ValueError(
                "Diagnostics require an existing task-owned report directory"
            )
        for path in (self.target, *self.target.parents):
            if path == root:
                break
            if path.is_symlink():
                raise ValueError("Symlink diagnostic directory")
        self.entries = []

    def __call__(self, diagnostic):
        if self.target.resolve() != self.target:
            raise ValueError("Diagnostic directory changed")
        if s.validate_manifest(self.manifest)["owner"] != self.owner:
            raise ValueError("Diagnostic owner changed")
        invocation = diagnostic.get("invocation_id", "")
        if not re.fullmatch("[0-9a-f]{32}", invocation):
            raise ValueError("Invalid diagnostic identity")
        raw = (
            json.dumps(diagnostic, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode()
        if len(raw) > 1_048_576:
            raise ValueError("Diagnostic payload too large")
        folder = self.target / "failed-responses"
        folder.mkdir(mode=0o700, exist_ok=True)
        if folder.is_symlink() or folder.stat().st_mode & 0o077:
            raise ValueError("Diagnostic directory is not private")
        path = folder / (invocation + ".json")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        self.entries.append(
            {
                k: diagnostic[k]
                for k in (
                    "invocation_id",
                    "input_hash",
                    "response_hash",
                    "model_step",
                    "stage",
                    "exception_type",
                    "error_code",
                    "issues",
                    "json_error",
                )
            }
        )
        for key in ("http_status", "provider_error", "replayable"):
            if key in diagnostic:
                self.entries[-1][key] = diagnostic[key]
