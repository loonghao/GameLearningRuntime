"""Build and run an isolated Unity Mono or IL2CPP provider sample."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--editor", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", choices=("mono", "il2cpp"), required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be a new or empty isolated project directory")
    root = Path(__file__).resolve().parents[1]
    assets = output / "Assets/GLR"
    (assets / "Editor").mkdir(parents=True)
    for source in (root / "sdk/csharp/GameLearningRuntime.Provider").glob("*.cs"):
        shutil.copy2(source, assets)
    for source in (root / "sdk/unity").glob("*.cs"):
        shutil.copy2(source, assets)
    shutil.copy2(root / "sdk/unity/Editor/ProviderSmoke.cs", assets / "Editor")
    method = "BuildMono" if args.backend == "mono" else "BuildIl2Cpp"
    subprocess.run(
        [
            str(args.editor.resolve()),
            "-batchmode",
            "-nographics",
            "-projectPath",
            str(output),
            "-executeMethod",
            f"ProviderSmoke.{method}",
            "-logFile",
            str(output / "build.log"),
        ],
        check=True,
        timeout=600,
    )
    player = output / "Builds" / ("Mono2x" if args.backend == "mono" else "IL2CPP")
    subprocess.run(
        [
            str(player / "GlrSmoke.exe"),
            "-batchmode",
            "-nographics",
            "--glr-smoke",
            "-logFile",
            str(output / "player.log"),
        ],
        check=True,
        timeout=60,
    )
    receipt = json.loads((player / "glr-player-smoke.json").read_text())
    if receipt != {
        "engine": "unity",
        "backend": args.backend,
        "steps": 3,
        "reset": True,
        "stale_rejected": True,
    }:
        raise ValueError("Unexpected Unity sample receipt")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
