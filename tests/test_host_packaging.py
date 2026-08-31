from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_host_packager_emits_a_deterministic_portable_archive(tmp_path: Path) -> None:
    binary = tmp_path / "glr-hostd.exe"
    binary.write_bytes(b"synthetic-host-binary")
    output = tmp_path / "release"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "package_host.py"),
            "--binary",
            str(binary),
            "--target",
            "x86_64-pc-windows-msvc",
            "--version",
            "0.2.0",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    archive = output / "glr-hostd-0.2.0-x86_64-pc-windows-msvc.zip"
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist() == ["glr-hostd.exe", "LICENSE", "README.md"]
        assert {entry.date_time for entry in bundle.infolist()} == {(1980, 1, 1, 0, 0, 0)}
        readme = bundle.read("README.md").decode("utf-8")
        assert "synthetic-counter" in readme
        assert "universal game loader" in readme
