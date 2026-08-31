from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_TARGETS = {
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "x86_64-pc-windows-msvc",
    "x86_64-unknown-linux-gnu",
}
VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")


def _entry(name: str, data: bytes, *, executable: bool) -> tuple[zipfile.ZipInfo, bytes]:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = ((0o755 if executable else 0o644) & 0xFFFF) << 16
    return info, data


def package_host(*, binary: Path, target: str, version: str, output: Path) -> Path:
    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"unsupported Rust target: {target}")
    if not VERSION.fullmatch(version):
        raise ValueError("version must be a semantic version without a v prefix")
    binary = binary.resolve()
    if not binary.is_file() or binary.is_symlink():
        raise ValueError("binary must be a regular non-linked file")
    license_path = ROOT / "LICENSE"
    readme_path = ROOT / "crates" / "glr-host" / "README.md"
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"glr-hostd-{version}-{target}.zip"
    executable_name = "glr-hostd.exe" if "windows" in target else "glr-hostd"
    entries = [
        _entry(executable_name, binary.read_bytes(), executable=True),
        _entry("LICENSE", license_path.read_bytes(), executable=False),
        _entry("README.md", readme_path.read_bytes(), executable=False),
    ]
    with zipfile.ZipFile(archive, "w") as bundle:
        for info, data in entries:
            bundle.writestr(info, data)
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description="Create one deterministic GLR Host archive")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--target", choices=sorted(SUPPORTED_TARGETS), required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, default=Path(".glr-release"))
    arguments = parser.parse_args()
    archive = package_host(
        binary=arguments.binary,
        target=arguments.target,
        version=arguments.version,
        output=arguments.output,
    )
    print(archive.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
