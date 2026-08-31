from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sdk" / "cpp" / "tests" / "provider_contract_smoke.cpp"
INCLUDE = ROOT / "sdk" / "cpp" / "include"


def _run_windows() -> None:
    program_files_x86 = os.environ.get("PROGRAMFILES(X86)")
    if not program_files_x86:
        raise RuntimeError("ProgramFiles(x86) is unavailable")
    vswhere = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        raise RuntimeError("Visual Studio locator is unavailable")
    installation = subprocess.run(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    vcvars = Path(installation) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not vcvars.is_file():
        raise RuntimeError("Visual C++ x64 environment is unavailable")
    command_shell = os.environ.get("COMSPEC", "cmd.exe")
    with tempfile.TemporaryDirectory(prefix="glr-cpp-sdk-") as temporary:
        object_path = Path(temporary) / "provider_contract_smoke.obj"
        command = (
            f'call "{vcvars}" >nul && cl /nologo /std:c++20 /EHsc /W4 /WX '
            f'/I"{INCLUDE}" /c "{SOURCE}" /Fo"{object_path}"'
        )
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            executable=command_shell,
            shell=True,
        )


def _run_posix() -> None:
    compiler = shutil.which("c++") or shutil.which("clang++")
    if compiler is None:
        raise RuntimeError("a C++20 compiler is unavailable")
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{INCLUDE}",
            "-fsyntax-only",
            str(SOURCE),
        ],
        cwd=ROOT,
        check=True,
    )


def main() -> int:
    if sys.platform == "win32":
        _run_windows()
    else:
        _run_posix()
    print("glr.host.v1 cpp-provider-sdk-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
