---
name: glr-qa
description: Run goal-driven GameLearningRuntime QA against an authorized game, training adapter, replay, or live probe and produce a dated JSON plus self-contained HTML report. Use when a player asks whether a game works well, wants bug discovery, regression checks, or evidence from bounded training runs.
license: MIT
---

# GLR QA

Turn a plain-language objective into bounded, inspectable QA evidence. Preserve the
boundary between deterministic checks, scripted replay, training metrics, and
live-host acceptance; a passing smoke command is not proof that the whole game is
complete.

1. Restate the goal and identify the authorized project/adapter and evidence scope.
2. Choose finite checks (for example adapter doctor, deterministic regression,
   replay, and an explicitly bounded training probe). Never invent credentials,
   game internals, or unrestricted automation.
3. Run the checks with `python -m game_learning_runtime.qa` or call
   `game_learning_runtime.qa.run_qa`. Use `--project` for the adapter working
   directory and one or more `--check NAME COMMAND...` arguments.
4. Inspect `result.json` and open `index.html` from the generated
   `.glr-qa/YYYY-MM-DD/<time>/` directory. Report failures with their command
   output, duration, and likely next investigation; report missing live evidence
   as an evidence gap.

Example:

```powershell
$env:PYTHONPATH = "src"
python -m game_learning_runtime.qa "inspect the whole game for bugs" `
  --project . `
  --check doctor glr --project . doctor `
  --check regression python -m pytest tests/test_runtime_integration.py -q `
  --check training python -m your_adapter.train --steps 1000
```

Do not claim release quality from this report alone. Keep proprietary traces and
secrets out of artifacts; publish only evidence the project owner authorized.
