from __future__ import annotations

import argparse
from pathlib import Path
from collections import deque

import numpy as np
from stable_baselines3 import PPO

from mario_runtime import build_rollout_summary
from mario_runtime import effective_goal_line_x
from mario_runtime import extract_sanitized_x_position
from mario_runtime import EnvConfig
from mario_runtime import load_env_config_for_model
from mario_runtime import make_single_env
from mario_runtime import render_rgb_frame
from mario_runtime import save_video
from mario_runtime import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Mario PPO model and record video.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=4_000)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fps", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    output_dir = Path(args.output_dir) if args.output_dir else model_path.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_env_config_for_model(
        model_path,
        fallback=EnvConfig(n_envs=1, noop_max=0, end_on_flag=True),
    )
    config.n_envs = 1
    config.noop_max = 0
    config.end_on_flag = True
    env = make_single_env(config, seed=args.seed)
    model = PPO.load(model_path, device="cpu")
    goal_x = effective_goal_line_x(config)

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    episode_flags: list[bool] = []
    max_x_positions: list[int] = []
    captured_frames = []
    frame_stack = deque(maxlen=config.frame_stack)

    for episode_idx in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode_idx)
        frame_stack.clear()
        for _ in range(config.frame_stack):
            frame_stack.append(obs)
        done = False
        episode_return = 0.0
        episode_length = 0
        episode_max_x = 0
        episode_x_pos = 0
        captured_frames.append(render_rgb_frame(env))

        while not done and episode_length < args.max_steps:
            stacked_obs = np.concatenate(list(frame_stack), axis=2).transpose(2, 0, 1)
            action, _ = model.predict(stacked_obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            frame_stack.append(obs)
            captured_frames.append(render_rgb_frame(env))
            episode_return += float(reward)
            episode_length += 1
            episode_x_pos = extract_sanitized_x_position(
                info, previous_x_pos=episode_x_pos, goal_line_x=goal_x
            )
            episode_max_x = max(episode_max_x, episode_x_pos)
            done = bool(terminated or truncated)

        episode_flags.append(bool(info.get("flag_get", False)))
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        max_x_positions.append(episode_max_x)

    video_path = output_dir / "evaluation.mp4"
    save_video(captured_frames, video_path, fps=args.fps)
    n_frames = len(captured_frames)
    playback_s = n_frames / max(args.fps, 1)

    summary = build_rollout_summary(
        episodes=args.episodes,
        deterministic=args.deterministic,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        episode_flags=episode_flags,
        max_x_positions=max_x_positions,
        video_path=str(video_path),
        video_fps=args.fps,
        video_num_frames=n_frames,
        goal_line_x=goal_x,
    )
    summary["model"] = str(model_path)
    write_json(summary, output_dir / "summary.json")
    print(
        "eval_ok "
        f"video={video_path} playback_s={playback_s:.3f} ({n_frames}f@{args.fps}fps) "
        f"avg_return={summary['average_return']:.2f} "
        f"flags_cleared={summary['flags_cleared']}/{args.episodes} "
        f"median_max_x={summary['median_max_x']:.1f}"
    )
    env.close()


if __name__ == "__main__":
    main()
