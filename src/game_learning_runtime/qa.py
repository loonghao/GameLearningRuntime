"""Goal-driven QA orchestration with dated, self-contained reports."""
from __future__ import annotations

import argparse
import html
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Sequence

QA_SCHEMA_VERSION = "glr.qa.v1"

@dataclass(frozen=True)
class QACheck:
    name: str
    status: str
    detail: str
    duration_seconds: float

@dataclass(frozen=True)
class QAResult:
    schema_version: str
    goal: str
    started_at: str
    finished_at: str
    output_dir: str
    checks: tuple[QACheck, ...]

    @property
    def status(self) -> str:
        return "passed" if all(c.status == "passed" for c in self.checks) else "failed"

def _run_check(name: str, command: Sequence[str], cwd: Path | None, timeout: int) -> QACheck:
    started = time.monotonic()
    try:
        completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return QACheck(name, "failed", str(exc), time.monotonic() - started)
    detail = (completed.stdout + "\n" + completed.stderr).strip()[-4000:]
    return QACheck(name, "passed" if completed.returncode == 0 else "failed", detail, time.monotonic() - started)

def render_html(result: QAResult) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(c.name)}</td><td class='{c.status}'>{c.status}</td>"
        f"<td>{c.duration_seconds:.2f}s</td><td><pre>{html.escape(c.detail)}</pre></td></tr>"
        for c in result.checks
    )
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>GLR QA report</title><style>body{{margin:0;background:#0b1020;color:#e8eefc;font:15px system-ui;padding:32px}}main{{max-width:1100px;margin:auto}}.card,table{{background:#121a2d;border:1px solid #273454;border-radius:14px;padding:18px}}.card{{display:inline-block;margin:6px;min-width:180px}}.value{{display:block;font-size:28px;font-weight:700;margin-top:6px}}table{{width:100%;border-collapse:separate;border-spacing:0;padding:0;overflow:hidden}}th,td{{padding:12px;border-bottom:1px solid #273454;text-align:left;vertical-align:top}}pre{{white-space:pre-wrap;color:#a9b8d8}}.passed{{color:#7ee2a8}}.failed{{color:#ff8e9e}}small{{color:#94a3c7}}</style></head><body><main><small>Game Learning Runtime · {QA_SCHEMA_VERSION}</small><h1>QA report</h1><p>{html.escape(result.goal)}</p><div class='card'>Status<span class='value {result.status}'>{result.status}</span></div><div class='card'>Checks<span class='value'>{len(result.checks)}</span></div><div class='card'>Passed<span class='value'>{sum(c.status == 'passed' for c in result.checks)}</span></div><p><small>{result.started_at} → {result.finished_at}</small></p><table><tr><th>Check</th><th>Status</th><th>Duration</th><th>Evidence</th></tr>{rows}</table></main></body></html>"""

def run_qa(goal: str, *, output_root: str | Path = ".glr-qa", project: str | Path | None = None, commands: Sequence[tuple[str, Sequence[str]]] = (), timeout: int = 900) -> QAResult:
    now = datetime.now(timezone.utc)
    day = now.astimezone().strftime("%Y-%m-%d")
    run_dir = Path(output_root) / day / now.strftime("%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    checks = tuple(_run_check(name, command, Path(project) if project else None, timeout) for name, command in commands)
    finished = datetime.now(timezone.utc)
    result = QAResult(QA_SCHEMA_VERSION, goal, now.isoformat(), finished.isoformat(), str(run_dir), checks)
    (run_dir / "result.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    (run_dir / "index.html").write_text(render_html(result), encoding="utf-8")
    return result

def main() -> int:
    parser = argparse.ArgumentParser(description="Run bounded GLR QA checks and build an HTML report")
    parser.add_argument("goal", help="human-readable QA objective")
    parser.add_argument("--output-root", default=".glr-qa")
    parser.add_argument("--project", default=None)
    parser.add_argument("--check", action="append", nargs="+", metavar="COMMAND", help="named check: NAME COMMAND...")
    args = parser.parse_args()
    commands = []
    for raw in args.check or []:
        if len(raw) < 2:
            parser.error("--check requires NAME COMMAND...")
        commands.append((raw[0], raw[1:]))
    result = run_qa(args.goal, output_root=args.output_root, project=args.project, commands=commands)
    print(json.dumps(asdict(result), indent=2))
    return 0 if result.status == "passed" else 1

if __name__ == "__main__":
    raise SystemExit(main())
