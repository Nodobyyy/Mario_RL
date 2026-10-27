from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from mario_runtime import timestamped_run_dir


PHASES = (
    {
        "name": "phase1_explore",
        "label": "Explore and stabilize forward progress",
        "ratio": 0.30,
        "action_set": "simple",
        "learning_rate": 2.5e-4,
        "n_steps": 512,
        "batch_size": 128,
        "ent_coef": 0.02,
        "forward_reward_scale": 0.20,
        "backward_penalty_scale": 0.05,
        "flag_bonus": 1500.0,
        "stall_steps": 64,
        "stall_penalty": 1.5,
    },
    {
        "name": "phase2_progress",
        "label": "Push consistent progress deeper into the level",
        "ratio": 0.40,
        "action_set": "simple",
        "learning_rate": 2.0e-4,
        "n_steps": 512,
        "batch_size": 128,
        "ent_coef": 0.01,
        "forward_reward_scale": 0.25,
        "backward_penalty_scale": 0.10,
        "flag_bonus": 2500.0,
        "stall_steps": 32,
        "stall_penalty": 4.0,
    },
    {
        "name": "phase3_finish",
        "label": "Bias toward level clears and faster finishes",
        "ratio": 0.30,
        "action_set": "simple",
        "learning_rate": 1.5e-4,
        "n_steps": 512,
        "batch_size": 128,
        "ent_coef": 0.005,
        "forward_reward_scale": 0.30,
        "backward_penalty_scale": 0.15,
        "flag_bonus": 4000.0,
        "stall_steps": 20,
        "stall_penalty": 8.0,
    },
)
LEVEL_1_1_GOAL_X = 3161
MAX_REASONABLE_X_POS = LEVEL_1_1_GOAL_X + 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a staged professional Mario PPO training pipeline."
    )
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--base-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=4_000)
    parser.add_argument("--preview-freq", type=int, default=50_000)
    parser.add_argument("--preview-steps", type=int, default=2_000)
    parser.add_argument("--preview-episodes", type=int, default=1)
    parser.add_argument("--preview-fps", type=int, default=15)
    parser.add_argument("--live-preview", action="store_true")
    return parser.parse_args()


def parse_seeds(seed_text: str) -> list[int]:
    seeds = []
    for raw in seed_text.split(","):
        raw = raw.strip()
        if raw:
            seeds.append(int(raw))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def allocate_phase_timesteps(total_timesteps: int) -> list[int]:
    raw = [max(1, int(total_timesteps * phase["ratio"])) for phase in PHASES]
    diff = total_timesteps - sum(raw)
    raw[-1] += diff
    return raw


def choose_resume_model(phase_dir: Path) -> Path:
    best_eval_model = phase_dir / "evaluations" / "best_eval.zip"
    best_eval_summary_path = phase_dir / "evaluations" / "best_eval_summary.json"
    if best_eval_model.exists() and best_eval_summary_path.exists():
        try:
            summary = load_json(best_eval_summary_path).get("summary") or {}
            clear_rate = float(summary.get("clear_rate", 0.0))
            median_max_x = float(summary.get("median_max_x", 0.0))
            average_max_x = float(summary.get("average_max_x", 0.0))
            best_max_x = float(summary.get("best_max_x", 0.0))
            looks_sane = (
                clear_rate > 0.0
                or (
                    0.0 <= median_max_x <= MAX_REASONABLE_X_POS
                    and 0.0 <= average_max_x <= MAX_REASONABLE_X_POS
                    and 0.0 <= best_max_x <= MAX_REASONABLE_X_POS
                )
            )
            if looks_sane:
                return best_eval_model
        except Exception:
            pass
    final_model = phase_dir / "models" / "mario_final.zip"
    if final_model.exists():
        return final_model
    if best_eval_model.exists():
        return best_eval_model
    return final_model


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_phase(
    *,
    project_dir: Path,
    phase: dict,
    timesteps: int,
    run_dir: Path,
    seed: int,
    device: str,
    n_envs: int,
    checkpoint_freq: int,
    eval_freq: int,
    eval_episodes: int,
    eval_steps: int,
    preview_freq: int,
    preview_steps: int,
    preview_episodes: int,
    preview_fps: int,
    live_preview: bool,
    resume_model: Path | None,
) -> None:
    command = [
        sys.executable,
        str(project_dir / "train_mario.py"),
        "--timesteps",
        str(timesteps),
        "--run-dir",
        str(run_dir),
        "--seed",
        str(seed),
        "--device",
        device,
        "--n-envs",
        str(n_envs),
        "--checkpoint-freq",
        str(checkpoint_freq),
        "--eval-freq",
        str(eval_freq),
        "--eval-episodes",
        str(eval_episodes),
        "--eval-steps",
        str(eval_steps),
        "--preview-freq",
        str(preview_freq),
        "--preview-steps",
        str(preview_steps),
        "--preview-episodes",
        str(preview_episodes),
        "--preview-fps",
        str(preview_fps),
        "--action-set",
        str(phase["action_set"]),
        "--learning-rate",
        str(phase["learning_rate"]),
        "--n-steps",
        str(phase["n_steps"]),
        "--batch-size",
        str(phase["batch_size"]),
        "--ent-coef",
        str(phase["ent_coef"]),
        "--forward-reward-scale",
        str(phase["forward_reward_scale"]),
        "--backward-penalty-scale",
        str(phase["backward_penalty_scale"]),
        "--flag-bonus",
        str(phase["flag_bonus"]),
        "--stall-steps",
        str(phase["stall_steps"]),
        "--stall-penalty",
        str(phase["stall_penalty"]),
    ]
    if resume_model:
        command.extend(["--resume-model", str(resume_model)])
    if live_preview:
        command.append("--live-preview")

    print(
        "phase_start "
        f"name={phase['name']} "
        f"timesteps={timesteps} "
        f"seed={seed} "
        f"resume={'yes' if resume_model else 'no'}"
    )
    subprocess.run(command, check=True, cwd=project_dir)


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    base_dir = (
        Path(args.base_dir)
        if args.base_dir
        else timestamped_run_dir(project_dir / "runs" / "professional")
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    phase_timesteps = allocate_phase_timesteps(args.total_timesteps)

    pipeline_summary = {
        "base_dir": str(base_dir),
        "project_dir": str(project_dir),
        "total_timesteps": args.total_timesteps,
        "seeds": seeds,
        "phases": [],
    }

    for seed in seeds:
        seed_dir = base_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        resume_model: Path | None = None
        seed_summary = {
            "seed": seed,
            "seed_dir": str(seed_dir),
            "phases": [],
        }

        for phase, phase_steps in zip(PHASES, phase_timesteps, strict=True):
            phase_dir = seed_dir / phase["name"]
            phase_dir.mkdir(parents=True, exist_ok=True)
            run_phase(
                project_dir=project_dir,
                phase=phase,
                timesteps=phase_steps,
                run_dir=phase_dir,
                seed=seed,
                device=args.device,
                n_envs=args.n_envs,
                checkpoint_freq=args.checkpoint_freq,
                eval_freq=args.eval_freq,
                eval_episodes=args.eval_episodes,
                eval_steps=args.eval_steps,
                preview_freq=args.preview_freq,
                preview_steps=args.preview_steps,
                preview_episodes=args.preview_episodes,
                preview_fps=args.preview_fps,
                live_preview=args.live_preview,
                resume_model=resume_model,
            )

            selected_model = choose_resume_model(phase_dir)
            eval_summary_path = phase_dir / "evaluations" / "best_eval_summary.json"
            phase_summary = {
                "name": phase["name"],
                "label": phase["label"],
                "run_dir": str(phase_dir),
                "timesteps": phase_steps,
                "selected_model": str(selected_model),
                "config": dict(phase),
            }
            if eval_summary_path.exists():
                phase_summary["best_eval"] = load_json(eval_summary_path)

            seed_summary["phases"].append(phase_summary)
            resume_model = selected_model

        pipeline_summary["phases"].append(seed_summary)
        (seed_dir / "professional_summary.json").write_text(
            json.dumps(seed_summary, indent=2),
            encoding="utf-8",
        )

    (base_dir / "professional_summary.json").write_text(
        json.dumps(pipeline_summary, indent=2),
        encoding="utf-8",
    )
    print(f"professional_train_ok base_dir={base_dir}")


if __name__ == "__main__":
    main()
