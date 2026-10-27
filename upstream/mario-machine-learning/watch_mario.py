from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from mario_runtime import EnvConfig
from mario_runtime import load_env_config_for_model
from mario_runtime import make_single_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a playable Mario viewer using a trained PPO model.")
    parser.add_argument(
        "--model",
        type=str,
        default=r"C:\Users\31660\mario-rl\runs\full-train\models\mario_final.zip",
    )
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    config = load_env_config_for_model(
        model_path,
        fallback=EnvConfig(n_envs=1, noop_max=0, end_on_flag=False),
    )
    config.n_envs = 1
    config.noop_max = 0
    config.end_on_flag = False
    env = make_single_env(config)
    model = PPO.load(model_path, device="cpu")

    obs, _ = env.reset()
    frame_stack = deque([obs] * config.frame_stack, maxlen=config.frame_stack)
    frame_delay = 1.0 / max(1, args.fps)

    print(f"viewer_started model={model_path}")
    print("Close the game window or stop the process when you're done.")

    for _ in range(args.max_steps):
        env.unwrapped.render(mode="human")
        stacked_obs = np.concatenate(list(frame_stack), axis=2).transpose(2, 0, 1)
        action, _ = model.predict(stacked_obs, deterministic=args.deterministic)
        obs, _, terminated, truncated, _ = env.step(int(action))
        frame_stack.append(obs)
        time.sleep(frame_delay)

        if terminated or truncated:
            obs, _ = env.reset()
            frame_stack.clear()
            for _ in range(config.frame_stack):
                frame_stack.append(obs)

    env.close()


if __name__ == "__main__":
    main()
