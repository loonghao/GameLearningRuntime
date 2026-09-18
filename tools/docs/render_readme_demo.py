"""Render the README GIF from a real synthetic GLR collection run."""

# ruff: noqa: E501 -- SVG template lines stay intact for visual maintenance.

from __future__ import annotations

import argparse
import html
import shutil
import subprocess
import tempfile
from pathlib import Path

from game_learning_runtime import ContractEnvironment, SyncCollector
from game_learning_runtime.examples import CounterEnvironment, always_increment

TARGET = 6
WIDTH = 960
HEIGHT = 540


def _render_frame(
    *,
    frame_index: int,
    frame_count: int,
    position: int,
    reward: float,
    total_reward: float,
    action: str,
    done: bool,
) -> str:
    progress = position / TARGET
    bar_width = round(678 * progress)
    status = "TERMINATED" if done else "RUNNING"
    status_color = "#67e8a5" if done else "#72a7ff"
    mask = "[true, false]" if done else "[true, true]"
    escaped_action = html.escape(action)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#09111f"/>
      <stop offset="1" stop-color="#121f36"/>
    </linearGradient>
    <linearGradient id="bar" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#5b8cff"/>
      <stop offset="1" stop-color="#63e6be"/>
    </linearGradient>
  </defs>
  <rect width="{WIDTH}" height="{HEIGHT}" rx="26" fill="url(#bg)"/>
  <text x="48" y="58" fill="#f7f9fc" font-family="Segoe UI, Arial, sans-serif" font-size="28" font-weight="700">Game Learning Runtime</text>
  <text x="48" y="87" fill="#8fa8c9" font-family="Segoe UI, Arial, sans-serif" font-size="15" letter-spacing="1.4">LOCAL RUN · SYNTHETIC ADAPTER · REAL CONTRACT DATA</text>
  <rect x="722" y="40" width="190" height="42" rx="21" fill="#172a45" stroke="{status_color}" stroke-width="1.5"/>
  <circle cx="747" cy="61" r="6" fill="{status_color}"/>
  <text x="764" y="67" fill="{status_color}" font-family="Consolas, monospace" font-size="15" font-weight="700">{status}</text>

  <rect x="48" y="116" width="864" height="122" rx="18" fill="#101c30" stroke="#263b58"/>
  <text x="78" y="150" fill="#8fa8c9" font-family="Segoe UI, Arial, sans-serif" font-size="14">Synthetic CounterEnvironment</text>
  <text x="78" y="184" fill="#f7f9fc" font-family="Segoe UI, Arial, sans-serif" font-size="26" font-weight="700">position {position} / {TARGET}</text>
  <rect x="78" y="202" width="678" height="12" rx="6" fill="#20314b"/>
  <rect x="78" y="202" width="{bar_width}" height="12" rx="6" fill="url(#bar)"/>
  <text x="788" y="210" fill="#b9c9dd" font-family="Consolas, monospace" font-size="14">frame {frame_index + 1}/{frame_count}</text>

  <rect x="48" y="258" width="864" height="90" rx="18" fill="#101c30" stroke="#263b58"/>
  <text x="78" y="287" fill="#8fa8c9" font-family="Segoe UI, Arial, sans-serif" font-size="13">RUNTIME ADAPTER</text>
  <text x="78" y="322" fill="#f7f9fc" font-family="Segoe UI, Arial, sans-serif" font-size="17">observation</text>
  <path d="M225 313 H302" stroke="#5b8cff" stroke-width="2"/>
  <path d="M296 307 L306 313 L296 319" fill="none" stroke="#5b8cff" stroke-width="2"/>
  <text x="330" y="287" fill="#8fa8c9" font-family="Segoe UI, Arial, sans-serif" font-size="13">GLR CONTRACT</text>
  <text x="330" y="322" fill="#f7f9fc" font-family="Segoe UI, Arial, sans-serif" font-size="17">validation PASS</text>
  <path d="M505 313 H582" stroke="#63e6be" stroke-width="2"/>
  <path d="M576 307 L586 313 L576 319" fill="none" stroke="#63e6be" stroke-width="2"/>
  <text x="610" y="287" fill="#8fa8c9" font-family="Segoe UI, Arial, sans-serif" font-size="13">SYNC COLLECTOR</text>
  <text x="610" y="322" fill="#f7f9fc" font-family="Segoe UI, Arial, sans-serif" font-size="17">transition accepted</text>

  <rect x="48" y="368" width="864" height="126" rx="18" fill="#07101d" stroke="#263b58"/>
  <text x="76" y="400" fill="#63e6be" font-family="Consolas, monospace" font-size="15">$ glr collect --stop-on-done</text>
  <text x="76" y="430" fill="#b9c9dd" font-family="Consolas, monospace" font-size="15">step={position}  action={escaped_action}  reward={reward:+.2f}</text>
  <text x="76" y="458" fill="#b9c9dd" font-family="Consolas, monospace" font-size="15">action_mask={mask}  total_reward={total_reward:+.2f}</text>
  <text x="76" y="482" fill="#7188a7" font-family="Consolas, monospace" font-size="13">No account, path, PID/HWND, or proprietary game data captured.</text>
</svg>
"""


def _collect_frames() -> list[tuple[int, float, float, str, bool]]:
    collector = SyncCollector(
        ContractEnvironment(CounterEnvironment(target=TARGET, max_steps=TARGET + 2)),
        actor_id="readme-demo",
    )
    unroll = collector.collect(always_increment, steps=TARGET + 2, seed=7, stop_on_done=True)

    frames: list[tuple[int, float, float, str, bool]] = [(0, 0.0, 0.0, "reset", False)]
    total_reward = 0.0
    for transition in unroll.transitions:
        position = int(transition.next_observation["position"][0])
        reward = float(transition.reward[0])
        total_reward += reward
        frames.append((position, reward, total_reward, "increment", transition.done))
    return frames


def render(output: Path) -> None:
    executable = shutil.which("magick")
    if executable is None:
        raise SystemExit("ImageMagick 'magick' is required to render the README GIF")

    frames = _collect_frames()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="glr-readme-demo-") as temporary_directory:
        directory = Path(temporary_directory)
        frame_paths: list[Path] = []
        for index, (position, reward, total_reward, action, done) in enumerate(frames):
            frame_path = directory / f"frame-{index:02d}.svg"
            frame_path.write_text(
                _render_frame(
                    frame_index=index,
                    frame_count=len(frames),
                    position=position,
                    reward=reward,
                    total_reward=total_reward,
                    action=action,
                    done=done,
                ),
                encoding="utf-8",
            )
            frame_paths.append(frame_path)

        command = [
            executable,
            "-background",
            "none",
            "-density",
            "96",
            "-delay",
            "70",
            *(str(path) for path in frame_paths[:-1]),
            "-delay",
            "170",
            str(frame_paths[-1]),
            "-loop",
            "0",
            "-layers",
            "Optimize",
            str(output),
        ]
        subprocess.run(command, check=True)

    print(f"rendered {len(frames)} verified frames to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/showcase/glr-counter-collector.gif"),
    )
    arguments = parser.parse_args()
    render(arguments.output)


if __name__ == "__main__":
    main()
