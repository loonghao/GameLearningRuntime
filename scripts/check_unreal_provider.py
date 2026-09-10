"""Run Actor, component and level checks in a new Unreal Editor project."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--editor", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be a new or empty isolated project directory")
    scripts = output / "Content/Python"
    scripts.mkdir(parents=True)
    for source in (Path(__file__).resolve().parents[1] / "sdk/unreal").glob("*.py"):
        shutil.copy2(source, scripts)
    project = output / "GlrSmoke.uproject"
    project.write_text(
        json.dumps(
            {
                "FileVersion": 3,
                "Plugins": [
                    {"Name": "PythonScriptPlugin", "Enabled": True},
                    {"Name": "EditorScriptingUtilities", "Enabled": True},
                ],
            }
        )
    )
    with (output / "startup.log").open("w") as log:
        subprocess.run(
            [
                str(args.editor.resolve()),
                str(project),
                "-unattended",
                "-nullrhi",
                "-nosplash",
                "-stdout",
                "-FullStdOutLogOutput",
                "-run=pythonscript",
                f"-script={scripts / 'smoke.py'}",
            ],
            stdout=log,
            stderr=log,
            check=True,
            timeout=300,
        )
    receipt = json.loads((output / "Saved/glr-smoke.json").read_text())
    if receipt != {
        "engine": "unreal-editor",
        "actor_steps": 3,
        "component_steps": 1,
        "level_transitions": 2,
        "reset": True,
        "stale_rejected": True,
    }:
        raise ValueError("Unexpected Unreal sample receipt")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
