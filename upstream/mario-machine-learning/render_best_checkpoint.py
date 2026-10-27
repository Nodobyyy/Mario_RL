"""
render_best_checkpoint.py
=========================

Take a trained PPO checkpoint and render the best evaluation episode as MP4
(and optionally GIF) for the README hero image.

The script tries N seeds × {deterministic, stochastic} and keeps one
episode (one MP4 = one rollout, no concatenation). Ranking is controlled
by ``--rank-by``:

* ``default``: among clears prefer higher max_x, then higher return; if none
  clear, prefer higher max_x then return.
* ``fastest_clear``: among clears prefer **fewest env steps** (shortest
  time-to-flag), then higher return; if none clear, same tie-break as
  ``default``.

Output:
    <output-dir>/best.mp4
    <output-dir>/best.gif   (optional, --gif)
    <output-dir>/best.json  (full episode summary)
    <output-dir>/all_runs.json  (summary of every attempted run)

Usage:
    # Render a specific checkpoint, try 5 seeds, also write GIF
    python render_best_checkpoint.py --model runs/ppo_overnight_TS/run/evaluations/best_eval.zip --seeds 5 --gif

    # Auto-detect newest run's best checkpoint
    python render_best_checkpoint.py --auto

    # Auto-detect, but prefer the latest periodic checkpoint over best_eval
    python render_best_checkpoint.py --auto --prefer-latest-step
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import cv2
from stable_baselines3 import PPO

from mario_runtime import EnvConfig
from mario_runtime import build_rollout_summary
from mario_runtime import effective_goal_line_x
from mario_runtime import extract_sanitized_x_position
from mario_runtime import load_env_config_for_model
from mario_runtime import make_single_env
from mario_runtime import render_rgb_frame
from mario_runtime import save_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Path to a PPO checkpoint .zip. Use --auto to skip and resolve automatically.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-detect newest checkpoint under runs/ppo_*",
    )
    parser.add_argument(
        "--prefer-latest-step",
        action="store_true",
        help="With --auto, prefer the newest periodic checkpoint over best_eval.zip.",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to try.")
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=4_000)
    parser.add_argument(
        "--policy-modes",
        type=str,
        default="deterministic,stochastic",
        help="Comma-separated subset of {deterministic,stochastic}.",
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--export-max-width",
        type=int,
        default=0,
        help="If >0, downscale every RGB frame so width<=this (height keeps aspect, INTER_AREA). 0=full NES res.",
    )
    parser.add_argument("--gif", action="store_true", help="Also write a GIF (slower, larger file).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Default: <model_dir>/best_render/",
    )
    parser.add_argument(
        "--rank-by",
        choices=("default", "fastest_clear"),
        default="default",
        help="How to pick the single episode to render (see module docstring).",
    )
    return parser.parse_args()


def auto_detect_checkpoint(prefer_latest_step: bool = False) -> Path | None:
    """Find the most relevant PPO checkpoint.

    Priority (default):
        1. Newest run dir's best_eval.zip
        2. Newest run dir's mario_final.zip
        3. Newest run dir's latest mario_ppo_<N>_steps.zip checkpoint

    With `prefer_latest_step=True`, the latest periodic checkpoint wins.
    Useful for inspecting an in-flight run where best_eval is stale.
    """
    runs_root = Path("runs")
    if not runs_root.exists():
        return None
    for run_dir in sorted(runs_root.glob("ppo_*"), reverse=True):
        latest_step = None
        ckpt_dir = run_dir / "run" / "models" / "checkpoints"
        if ckpt_dir.exists():
            steps = sorted(
                ckpt_dir.glob("mario_ppo_*_steps.zip"),
                key=lambda p: int(p.stem.split("_")[-2]),
                reverse=True,
            )
            if steps:
                latest_step = steps[0]

        candidates_in_order = (
            [latest_step] if prefer_latest_step and latest_step else []
        ) + [
            run_dir / "run" / "evaluations" / "best_eval.zip",
            run_dir / "run" / "models" / "mario_final.zip",
        ]
        if not prefer_latest_step and latest_step:
            candidates_in_order.append(latest_step)

        for path in candidates_in_order:
            if path and path.exists():
                return path
    return None


def episode_quality_key_default(ep: dict) -> tuple[int, int, float]:
    return (1 if ep["flag"] else 0, ep["max_x"], ep["return"])


def episode_quality_key_fastest_clear(ep: dict) -> tuple[int, int, float]:
    """Prefer flag clears with minimum step count; shorter length -> larger key."""
    if ep["flag"]:
        return (2, -int(ep["length"]), int(round(ep["return"] * 1000)))
    return (1, ep["max_x"], int(round(ep["return"] * 1000)))


def select_quality_key(rank_by: str):
    if rank_by == "fastest_clear":
        return episode_quality_key_fastest_clear
    return episode_quality_key_default


def _set_rollout_seeds(seed: int) -> None:
    """Make a single stochastic rollout reproducible for a given env seed (CPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_one(
    *,
    model: PPO,
    config: EnvConfig,
    seed: int,
    deterministic: bool,
    max_steps: int,
) -> dict:
    _set_rollout_seeds(seed)
    env = make_single_env(config, seed=seed)
    obs, _info = env.reset(seed=seed)
    frame_stack: deque[np.ndarray] = deque(maxlen=config.frame_stack)
    frame_stack.clear()
    for _ in range(config.frame_stack):
        frame_stack.append(obs)

    frames: list[np.ndarray] = [render_rgb_frame(env)]
    episode_return = 0.0
    episode_length = 0
    episode_max_x = 0
    episode_x_pos = 0
    flag = False
    done = False
    goal_x = effective_goal_line_x(config)

    while not done and episode_length < max_steps:
        stacked = np.concatenate(list(frame_stack), axis=2).transpose(2, 0, 1)
        action, _ = model.predict(stacked, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        frame_stack.append(obs)
        frames.append(render_rgb_frame(env))
        episode_return += float(reward)
        episode_length += 1
        episode_x_pos = extract_sanitized_x_position(info, previous_x_pos=episode_x_pos)
        episode_max_x = max(episode_max_x, episode_x_pos)
        if info.get("flag_get", False):
            flag = True
        done = bool(terminated or truncated)

    env.close()
    return {
        "seed": seed,
        "deterministic": deterministic,
        "return": episode_return,
        "length": episode_length,
        "max_x": int(episode_max_x),
        "flag": bool(flag),
        "frames": frames,
    }


def downscale_frame_rgb(frame: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    new_w = max_width
    new_h = max(1, int(round(h * (max_width / w))))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def downscale_frames(frames: list[np.ndarray], max_width: int) -> list[np.ndarray]:
    if max_width <= 0:
        return frames
    return [downscale_frame_rgb(f, max_width) for f in frames]


def write_gif(frames: list[np.ndarray], path: Path, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError:
        print(f"  imageio not available; skipping GIF write to {path}")
        return
    duration_s = max(1.0 / fps, 0.01)
    imageio.mimwrite(str(path), frames, format="GIF", duration=duration_s, loop=0)


def main() -> None:
    args = parse_args()

    if args.auto and not args.model:
        detected = auto_detect_checkpoint(prefer_latest_step=args.prefer_latest_step)
        if not detected:
            raise SystemExit("No PPO checkpoint found under runs/. Pass --model explicitly.")
        model_path = detected
        print(f"auto_detected_model={model_path}")
    elif args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(model_path)
    else:
        raise SystemExit("Pass either --model <path> or --auto.")

    output_dir = Path(args.output_dir) if args.output_dir else model_path.parent / "best_render"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_env_config_for_model(
        model_path,
        fallback=EnvConfig(n_envs=1, noop_max=0, end_on_flag=True),
    )
    config.n_envs = 1
    config.noop_max = 0
    config.end_on_flag = True

    model = PPO.load(model_path, device="cpu")

    modes = []
    for mode_token in args.policy_modes.split(","):
        mode_token = mode_token.strip().lower()
        if mode_token == "deterministic":
            modes.append(True)
        elif mode_token == "stochastic":
            modes.append(False)
    if not modes:
        modes = [True]

    all_runs: list[dict] = []
    best: dict | None = None
    qkey = select_quality_key(args.rank_by)

    for seed_idx in range(args.seeds):
        for deterministic in modes:
            seed = args.seed_base + seed_idx
            print(f"  rolling out seed={seed} deterministic={deterministic} ...")
            ep = run_one(
                model=model,
                config=config,
                seed=seed,
                deterministic=deterministic,
                max_steps=args.max_steps,
            )
            all_runs.append({k: v for k, v in ep.items() if k != "frames"})
            if best is None or qkey(ep) > qkey(best):
                best = ep
            print(
                f"    -> return={ep['return']:.1f}  max_x={ep['max_x']}  "
                f"len={ep['length']}  flag={ep['flag']}"
            )

    assert best is not None
    print(
        f"BEST rank_by={args.rank_by} seed={best['seed']} "
        f"deterministic={best['deterministic']} max_x={best['max_x']} "
        f"flag={best['flag']} len={best['length']} return={best['return']:.1f}"
    )

    out_frames = downscale_frames(best["frames"], args.export_max_width)
    if args.export_max_width > 0:
        print(f"  export: downscaled to max_width={args.export_max_width} px (cv2.INTER_AREA)")

    mp4_path = output_dir / "best.mp4"
    save_video(out_frames, mp4_path, fps=args.fps)
    n_frames = len(out_frames)
    playback_s = n_frames / max(args.fps, 1)
    print(
        f"  wrote {mp4_path}  ({n_frames} frames @ {args.fps} fps -> playback {playback_s:.3f}s)"
    )

    if args.gif:
        gif_path = output_dir / "best.gif"
        write_gif(out_frames, gif_path, fps=args.fps)
        if gif_path.exists():
            print(f"  wrote {gif_path}")

    summary = build_rollout_summary(
        episodes=1,
        deterministic=best["deterministic"],
        episode_returns=[best["return"]],
        episode_lengths=[best["length"]],
        episode_flags=[best["flag"]],
        max_x_positions=[best["max_x"]],
        video_path=str(mp4_path),
        video_fps=args.fps,
        video_num_frames=n_frames,
        goal_line_x=effective_goal_line_x(config),
    )
    summary["seed"] = best["seed"]
    summary["model"] = str(model_path)
    summary["rank_by"] = args.rank_by
    (output_dir / "best.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    (output_dir / "all_runs.json").write_text(
        json.dumps(all_runs, indent=2),
        encoding="utf-8",
    )
    print(f"  wrote {output_dir / 'all_runs.json'}")


if __name__ == "__main__":
    main()
