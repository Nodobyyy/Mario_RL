from __future__ import annotations

import os

import gymnasium as gym
from gymnasium_super_mario_bros.actions import SIMPLE_MOVEMENT
from gymnasium_super_mario_bros.smb_env import SuperMarioBrosEnv
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


class LegacyMarioToGymnasium(gym.Wrapper):
    """Adapt the Mario env's legacy Gym API to Gymnasium's reset/step signatures."""

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None and hasattr(self.env, "seed"):
            self.env.seed(seed)
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = bool(done)
        truncated = False
        return obs, reward, terminated, truncated, info


def make_env(render_mode: str | None = None):
    def _factory():
        env = SuperMarioBrosEnv(target=(1, 1), render_mode=render_mode)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = LegacyMarioToGymnasium(env)
        env = Monitor(env)
        env = WarpFrame(env)
        return env

    return _factory


if __name__ == "__main__":
    total_timesteps = int(os.environ.get("MARIO_TIMESTEPS", "2048"))
    model_path = os.environ.get("MARIO_MODEL_PATH", "mario_smoke_test.zip")

    env = DummyVecEnv([make_env()])
    env = VecFrameStack(env, n_stack=4)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=256,
        batch_size=64,
        learning_rate=2.5e-4,
        device="cpu",
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(model_path)

    obs = env.reset()
    for _ in range(64):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        if bool(dones[0]):
            obs = env.reset()

    print(f"smoke_ok saved={model_path} timesteps={total_timesteps}")
    env.close()
