"""Validate that GLR can drive a browser game and learn to play it.

This is the evidence producer for the browser/web-game capability question. It
serves the bundled Three.js page over loopback HTTP, wraps it in a
:class:`~game_learning_runtime.environment.ContractEnvironment` adapter,
collects unrolls with
:class:`~game_learning_runtime.collector.SyncCollector`, trains a small PPO
policy on those unrolls, and reports the before/after episode return.

The comparison is the point of the tool. "It connects" and "it steps" are not
evidence that the framework can make an agent play well, so this tool measures
a random policy and the trained policy on the same seeds and reports both.

Seeding is explicit end to end. ``--seed`` pins the policy RNGs *and* the game
world: every ``reset()`` sinks a per-episode seed derived from it, because an
adapter that reseeds itself when the caller passes no seed would otherwise give
every episode a different world and the comparison would measure the sampling
instead of the learning. Two runs with the same seed therefore produce
byte-identical reports; without per-episode seeding they do not.

Usage:

```bash
python tools/providers/validate_web_rl.py --output artifacts/web-rl-validation.json
```

Requires the optional ``playwright`` and ``torch`` dependencies and a Chromium
browser. Run ``playwright install chromium`` first if the browser is missing.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from game_learning_runtime.collector import SyncCollector
from game_learning_runtime.contracts import TimeStep, Transition
from game_learning_runtime.environment import ContractEnvironment, GameEnvironment
from game_learning_runtime.specs import CompositeSpec
from game_learning_runtime.web_game.environments import DODGE_ACTIONS
from game_learning_runtime.web_game.serving import LocalPageServer, bundled_assets_dir

PAGE_NAME = "orbital_dodge.html"

# Evaluation episodes draw their world seeds from a reserved range so the worlds
# a policy is scored on are never worlds it trained on. A 120k-step run completes
# a few thousand training episodes; the offset keeps the two ranges disjoint.
EVALUATION_SEED_OFFSET = 1_000_000

# Frozen record of the run this tool produced before the seed was sunk into every
# reset. It travels in the report so a later reader can see exactly what the
# per-episode seeding changed instead of having to re-run an old commit.
BEFORE_SEED_SINKING: dict[str, Any] = {
    "produced_by": "tools/providers/validate_web_rl.py at main 5291135",
    "world_seed": "none: every reset() drew a fresh uuid4() world seed",
    "config": {
        "train_steps": 120_000,
        "evaluation_episodes": 30,
        "max_steps": 256,
        "unroll_length": 128,
        "seed": 7,
    },
    "baseline_random_policy": {
        "episodes": 30.0,
        "mean_steps": 65.83333333333333,
        "std_steps": 41.90153802533851,
        "mean_reward": 0.7331998183391988,
        "best_reward": 4.702166482806206,
    },
    "trained_policy": {
        "episodes": 30.0,
        "mean_steps": 103.86666666666666,
        "std_steps": 76.38836444264416,
        "mean_reward": 2.309140901329617,
        "best_reward": 9.420000003650784,
    },
    "improvement": {
        "mean_steps_ratio": 1.5777215189873417,
        "mean_reward_ratio": 3.1494019005080327,
    },
    "policy_updates": 7500,
}


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """One evaluated episode: how long it lasted and what it earned."""

    steps: int
    reward: float
    score: float


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Bounded budgets for one validation run."""

    max_steps: int = 256
    frames_per_step: int = 1
    train_steps: int = 20_000
    unroll_length: int = 128
    epochs: int = 4
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    evaluation_episodes: int = 8
    seed: int = 0
    headless: bool = True

    def __post_init__(self) -> None:
        for name in ("max_steps", "train_steps", "unroll_length", "epochs", "minibatch_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.evaluation_episodes < 1:
            raise ValueError("evaluation_episodes must be a positive integer")


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "validate_web_rl requires the optional torch dependency: pip install '.[torch]'"
        ) from error
    return torch


def build_environment(config: ValidationConfig) -> GameEnvironment:
    """Serve the bundled page and return a contract-checked adapter over it."""

    try:
        from game_learning_runtime.web_game.bridge import PageSpec, PlaywrightBrowserBridge
        from game_learning_runtime.web_game.environments import InstrumentedWebGameEnvironment
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise SystemExit(f"validate_web_rl requires playwright: {error}") from error

    server = LocalPageServer(bundled_assets_dir())
    server.start()
    page_spec = PageSpec(
        url=server.url_for(PAGE_NAME),
        ready_expression="window.__glr && window.__glr.ready === true",
        reset_expression="window.__glr.reset(0)",
    )
    bridge = PlaywrightBrowserBridge(page_spec, headless=config.headless)
    return InstrumentedWebGameEnvironment(
        bridge,
        max_steps=config.max_steps,
        frames_per_step=config.frames_per_step,
        close_bridge=True,
    )


class PPOTrainer:
    """Minimal discrete-action PPO over unrolls collected by GLR.

    The learner is deliberately small and lives in ``tools/``: the runtime ships
    objectives, not learners (ADR-0005), so the reusable part -- advantage
    estimation and the clipped objective -- comes from
    :mod:`game_learning_runtime.integrations.torch_objectives`. This class is
    only the update loop that feeds them.
    """

    def __init__(
        self,
        *,
        observation_spec: CompositeSpec,
        action_count: int,
        config: ValidationConfig,
    ) -> None:
        torch = _require_torch()
        self._torch = torch
        self._config = config
        self._action_count = action_count
        feature_count = int(np.prod(_leaf_shape(observation_spec)))
        torch.manual_seed(config.seed)
        # A shared trunk with one hidden layer. The bundled task is not linearly
        # separable -- identical player positions need different actions
        # depending on where the threat is -- so a linear policy plateaus early
        # and a wider net would only slow a validation run down.
        self._trunk = torch.nn.Sequential(
            torch.nn.Linear(feature_count, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
        )
        self._policy_head = torch.nn.Linear(64, action_count)
        self._value_head = torch.nn.Linear(64, 1)
        torch.nn.init.orthogonal_(self._policy_head.weight, gain=0.01)
        torch.nn.init.zeros_(self._policy_head.bias)
        torch.nn.init.orthogonal_(self._value_head.weight, gain=1.0)
        torch.nn.init.zeros_(self._value_head.bias)
        self._optimizer = torch.optim.Adam(self._parameters(), lr=config.learning_rate)
        self.updates = 0

    def _parameters(self) -> list[Any]:
        return (
            list(self._trunk.parameters())
            + list(self._policy_head.parameters())
            + list(self._value_head.parameters())
        )

    def features(self, timestep: TimeStep) -> Any:
        """Flatten one observation into the network's input vector."""

        torch = self._torch
        # Contract observations are frozen read-only arrays; torch needs a copy.
        values = np.array(timestep.observation["features"], dtype=np.float32).reshape(-1)
        return torch.from_numpy(values)

    def act(self, timestep: TimeStep, *, deterministic: bool = False) -> dict[str, Any]:
        """Sample (or take the argmax of) one action for a time step."""

        torch = self._torch
        with torch.no_grad():
            logits = self._policy_head(self._trunk(self.features(timestep)))
            mask = timestep.action_mask
            if mask is not None:
                allowed = torch.from_numpy(np.array(mask["choice"], dtype=np.bool_).reshape(-1))
                logits = logits.masked_fill(~allowed, -1e9)
            if deterministic:
                index = int(torch.argmax(logits).item())
            else:
                index = int(torch.distributions.Categorical(logits=logits).sample().item())
            value = self._value_head(self._trunk(self.features(timestep))).squeeze(-1)
        return {
            "choice": np.array([index], dtype=np.int64),
            "_logits": logits,
            "_index": index,
            "_value": value,
        }

    def remember(self, transition: Transition, decision: dict[str, Any]) -> None:
        """Record the per-step statistics one PPO update needs."""

        self._batch.append(
            {
                "features": self.features(_as_timestep(transition)),
                "action": decision["_index"],
                "log_prob": float(
                    self._torch.distributions.Categorical(logits=decision["_logits"]).log_prob(
                        self._torch.tensor(decision["_index"])
                    )
                ),
                "value": float(decision["_value"]),
                "reward": float(np.asarray(transition.reward).reshape(-1)[0]),
                "terminated": bool(np.asarray(transition.terminated).reshape(-1)[0]),
                "truncated": bool(np.asarray(transition.truncated).reshape(-1)[0]),
            }
        )

    def start_batch(self) -> None:
        self._batch: list[dict[str, Any]] = []

    def has_pending_batch(self) -> bool:
        """Whether unflushed transitions are waiting for an update."""

        return bool(self._batch)

    def update(self) -> dict[str, float]:
        """Run PPO epochs over the current batch and return training metrics."""

        from game_learning_runtime.integrations.torch_objectives import (
            generalized_advantage_estimate,
            ppo_loss,
        )

        torch = self._torch
        if not self._batch:
            return {}
        rewards = torch.tensor([row["reward"] for row in self._batch], dtype=torch.float32)
        values = torch.tensor([row["value"] for row in self._batch] + [0.0], dtype=torch.float32)
        terminated = torch.tensor([row["terminated"] for row in self._batch])
        truncated = torch.tensor([row["truncated"] for row in self._batch])
        targets = generalized_advantage_estimate(
            rewards=rewards,
            values=values,
            terminated=terminated,
            truncated=truncated,
            gamma=self._config.gamma,
            gae_lambda=self._config.gae_lambda,
        )
        features = torch.stack([row["features"] for row in self._batch])
        actions = torch.tensor([row["action"] for row in self._batch], dtype=torch.int64)
        old_log_prob = torch.tensor([row["log_prob"] for row in self._batch], dtype=torch.float32)
        old_values = values[:-1].detach()

        metrics: dict[str, float] = {}
        indices = torch.arange(features.shape[0])
        for _ in range(self._config.epochs):
            order = indices[torch.randperm(indices.shape[0])]
            for start in range(0, order.shape[0], self._config.minibatch_size):
                batch = order[start : start + self._config.minibatch_size]
                # The trunk runs inside the epoch loop: a graph built once and
                # reused across epochs is freed by the first backward pass.
                hidden = self._trunk(features[batch])
                loss = ppo_loss(
                    policy_logits=self._policy_head(hidden),
                    actions=actions[batch],
                    old_log_prob=old_log_prob[batch],
                    advantages=targets.advantages[batch],
                    values=self._value_head(hidden).squeeze(-1),
                    value_targets=targets.value_targets[batch],
                    old_values=old_values[batch],
                    clip_epsilon=self._config.clip_epsilon,
                    value_clip_epsilon=self._config.clip_epsilon,
                    value_coefficient=self._config.value_coefficient,
                    entropy_coefficient=self._config.entropy_coefficient,
                )
                self._optimizer.zero_grad(set_to_none=True)
                loss.loss.backward()
                torch.nn.utils.clip_grad_norm_(self._parameters(), 0.5)
                self._optimizer.step()
                self.updates += 1
                metrics = {
                    "policy_loss": float(loss.policy_loss.detach()),
                    "value_loss": float(loss.value_loss.detach()),
                    "entropy": float(loss.entropy.detach()),
                    "approximate_kl": float(loss.approximate_kl.detach()),
                }
        self._batch = []
        return metrics


def _ratio(trained: float, baseline: float) -> float:
    """Return a guarded ratio so a near-zero baseline cannot blow up the report."""

    if abs(baseline) < 1e-9:
        return float("nan") if abs(trained) < 1e-9 else float("inf")
    return trained / baseline


def _leaf_shape(spec: CompositeSpec) -> tuple[int, ...]:
    leaves = spec.flatten()
    if len(leaves) != 1:
        raise ValueError(f"expected one observation leaf; received {sorted(leaves)}")
    shape = next(iter(leaves.values())).shape
    return tuple(dimension for dimension in shape if dimension is not None)


def _as_timestep(transition: Transition) -> TimeStep:
    """Rebuild the pre-action time step a transition was collected from."""

    return TimeStep(
        observation=transition.observation,
        reward=transition.reward,
        terminated=transition.terminated,
        truncated=transition.truncated,
        action_mask=transition.action_mask,
        episode_id=transition.episode_id,
        step_id=transition.step_id,
    )


def episode_seed(base: int, episode_index: int) -> int:
    """Return the world seed for one episode of a seeded validation run.

    Episode ``i`` is reset with ``base + i``. Passing the seed on is what makes a
    run reproducible: an adapter that reseeds itself when the caller passes no
    seed would replay a different world on every episode, so the same command
    would report different numbers twice.
    """

    return base + episode_index


def evaluate(
    environment: GameEnvironment,
    policy: Callable[[TimeStep], dict[str, Any]],
    *,
    episodes: int,
    max_steps: int,
    seed: int,
) -> list[EpisodeResult]:
    """Run whole episodes and report length, return, and in-game score.

    ``seed`` is the world seed of episode zero; the random and the trained policy
    are evaluated with the same seed so they play the same worlds.
    """

    results: list[EpisodeResult] = []
    for index in range(episodes):
        timestep = environment.reset(seed=episode_seed(seed, index))
        total_reward = 0.0
        steps = 0
        score = 0.0
        while not timestep.done and steps < max_steps:
            action = policy(timestep)
            timestep = environment.step({"choice": action["choice"]})
            total_reward += float(np.asarray(timestep.reward).reshape(-1)[0])
            score = float(timestep.info.get("score", score))
            steps += 1
        results.append(EpisodeResult(steps=steps, reward=total_reward, score=score))
    return results


def summarize(results: Sequence[EpisodeResult]) -> dict[str, float]:
    """Return mean/std aggregates for a set of evaluated episodes."""

    rewards = np.array([result.reward for result in results], dtype=np.float64)
    steps = np.array([result.steps for result in results], dtype=np.float64)
    scores = np.array([result.score for result in results], dtype=np.float64)
    return {
        "episodes": float(len(results)),
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "mean_steps": float(steps.mean()),
        "std_steps": float(steps.std()),
        "mean_score": float(scores.mean()),
        "best_reward": float(rewards.max()),
    }


def run_validation(config: ValidationConfig) -> dict[str, Any]:
    """Collect, train, and compare; return the full evidence payload."""

    environment = ContractEnvironment(build_environment(config))
    collector = SyncCollector(environment, actor_id="web-ppo")
    trainer = PPOTrainer(
        observation_spec=environment.spec.observation,
        action_count=len(DODGE_ACTIONS),
        config=config,
    )
    try:
        rng = np.random.default_rng(config.seed)

        def random_policy(timestep: TimeStep) -> dict[str, Any]:
            del timestep
            return {"choice": rng.integers(0, len(DODGE_ACTIONS), size=1, dtype=np.int64)}

        # Both policies are scored on the same held-out worlds, so the
        # difference between them is the policy and not the sample.
        evaluation_seed = config.seed + EVALUATION_SEED_OFFSET
        baseline = evaluate(
            environment,
            random_policy,
            episodes=config.evaluation_episodes,
            max_steps=config.max_steps,
            seed=evaluation_seed,
        )

        curve: list[dict[str, object]] = []
        collected = 0
        last_metrics: dict[str, float] = {}
        probe_interval = max(1, config.train_steps // 50)
        window_episodes: list[int] = []
        completed_episodes = 0
        trainer.start_batch()
        timestep = environment.reset(seed=episode_seed(config.seed, completed_episodes))
        while collected < config.train_steps:
            decision = trainer.act(timestep)
            action = {"choice": decision["choice"]}
            previous = timestep
            timestep = environment.step(action)
            transition = Transition(
                episode_id=previous.episode_id,
                step_id=timestep.step_id,
                observation=previous.observation,
                action=action,
                reward=timestep.reward,
                next_observation=timestep.observation,
                terminated=timestep.terminated,
                truncated=timestep.truncated,
                action_mask=previous.action_mask,
                next_action_mask=timestep.action_mask,
            )
            trainer.remember(transition, decision)
            collected += 1
            if timestep.done:
                # Every completed training episode is a free sample of how the
                # policy is doing. A window of fifty says far more than two probe
                # episodes do, and it costs no extra environment steps.
                window_episodes.append(int(timestep.info.get("web_steps", 0)))
                if len(window_episodes) > 50:
                    window_episodes.pop(0)
                completed_episodes += 1
                timestep = environment.reset(seed=episode_seed(config.seed, completed_episodes))
            if collected % config.unroll_length == 0:
                last_metrics = trainer.update()
                trainer.start_batch()
            if collected % probe_interval == 0 and window_episodes:
                entry = {
                    "collected_steps": float(collected),
                    "mean_steps": float(np.mean(window_episodes)),
                    "std_steps": float(np.std(window_episodes)),
                    "episode_samples": float(len(window_episodes)),
                }
                curve.append({"phase": "training", **entry})
                print(json.dumps({"phase": "training", **entry}, sort_keys=True), flush=True)
        if trainer.has_pending_batch():
            last_metrics = trainer.update()

        trained = evaluate(
            environment,
            lambda ts: trainer.act(ts, deterministic=True),
            episodes=config.evaluation_episodes,
            max_steps=config.max_steps,
            seed=evaluation_seed,
        )
        curve.append(
            {
                "phase": "trained",
                "collected_steps": float(collected),
                **summarize(trained),
                **last_metrics,
            }
        )
        baseline_summary = summarize(baseline)
        trained_summary = summarize(trained)
        return {
            "environment_id": environment.spec.environment_id,
            "config": asdict(config),
            "seeding": {
                "base_seed": config.seed,
                "world_seed": "per episode, sunk into GameEnvironment.reset(seed=...)",
                "training_episode_seed": "base_seed + completed-episode index",
                "training_episodes": float(completed_episodes),
                "evaluation_episode_seed": (
                    f"base_seed + {EVALUATION_SEED_OFFSET} + episode index"
                ),
                "evaluation_seeds": [
                    episode_seed(evaluation_seed, index)
                    for index in range(config.evaluation_episodes)
                ],
                "evaluation_worlds": "held out from training and shared by both policies",
            },
            "baseline_random_policy": baseline_summary,
            "trained_policy": trained_summary,
            "improvement": {
                "mean_reward_ratio": _ratio(
                    trained_summary["mean_reward"], baseline_summary["mean_reward"]
                ),
                "mean_steps_ratio": _ratio(
                    trained_summary["mean_steps"], baseline_summary["mean_steps"]
                ),
                "mean_reward_delta": trained_summary["mean_reward"]
                - baseline_summary["mean_reward"],
                "mean_steps_delta": trained_summary["mean_steps"] - baseline_summary["mean_steps"],
            },
            "curve": curve,
            "policy_updates": trainer.updates,
            "history": {"before_seed_sinking": BEFORE_SEED_SINKING},
        }
    finally:
        environment.close()
        collector  # noqa: B018 - the collector owns no resources beyond the environment


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    # Defaults come from a constructed instance: a slotted dataclass exposes its
    # fields as descriptors on the class, so `ValidationConfig.seed` is not 0.
    defaults = ValidationConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True, help="Where to write the JSON report")
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--train-steps", type=int, default=defaults.train_steps)
    parser.add_argument("--unroll-length", type=int, default=defaults.unroll_length)
    parser.add_argument("--evaluation-episodes", type=int, default=defaults.evaluation_episodes)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--headed", action="store_true", help="Show the browser window")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = ValidationConfig(
        max_steps=args.max_steps,
        train_steps=args.train_steps,
        unroll_length=args.unroll_length,
        evaluation_episodes=args.evaluation_episodes,
        seed=args.seed,
        headless=not args.headed,
    )
    report = run_validation(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
