"""Random Network Distillation (RND) intrinsic reward for Mario PPO.

Implements the core ideas from Burda et al. "Exploration by Random Network Distillation".
Applied as a VecEnvWrapper: observations are passed through two small CNNs
(a frozen random target and a trainable predictor). The MSE between their feature
vectors is normalized and added to the extrinsic reward. The predictor is updated
online using every batch of observations so that seen states quickly yield low
intrinsic reward while genuinely novel states (e.g. past the tall pipe) spike.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


class _SmallCnn(nn.Module):
    def __init__(self, in_channels: int, feature_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDModule:
    """Trainable RND predictor paired with a frozen random target network."""

    def __init__(
        self,
        *,
        obs_shape: tuple[int, int, int],
        device: str = "cuda",
        feature_dim: int = 256,
        lr: float = 1e-4,
        clip_obs: float = 5.0,
        reward_clip: float = 5.0,
    ) -> None:
        self.device = torch.device(device)
        self.clip_obs = clip_obs
        self.reward_clip = reward_clip

        in_channels = obs_shape[0]
        self.target = _SmallCnn(in_channels, feature_dim).to(self.device)
        self.predictor = _SmallCnn(in_channels, feature_dim).to(self.device)
        for p in self.target.parameters():
            p.requires_grad = False
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.obs_rms = RunningMeanStd(shape=obs_shape)
        self.int_rew_rms = RunningMeanStd(shape=())

    def compute_and_train(self, obs_np: np.ndarray) -> np.ndarray:
        """Return normalized intrinsic reward per env and take one optimizer step.

        Expects obs shaped (N, C, H, W) with dtype uint8.
        """
        obs_f = obs_np.astype(np.float32) / 255.0
        self.obs_rms.update(obs_f)

        mean = torch.as_tensor(self.obs_rms.mean, device=self.device, dtype=torch.float32)
        var = torch.as_tensor(self.obs_rms.var, device=self.device, dtype=torch.float32)
        obs_t = torch.as_tensor(obs_f, device=self.device, dtype=torch.float32)
        obs_norm = (obs_t - mean) / torch.sqrt(var + 1e-5)
        obs_norm = torch.clamp(obs_norm, -self.clip_obs, self.clip_obs)

        with torch.no_grad():
            target_feat = self.target(obs_norm)
        pred_feat = self.predictor(obs_norm)
        err = (pred_feat - target_feat).pow(2).mean(dim=-1)
        loss = err.mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        self.optimizer.step()

        intrinsic = err.detach().cpu().numpy().astype(np.float32)
        self.int_rew_rms.update(intrinsic)
        norm = intrinsic / (np.sqrt(self.int_rew_rms.var) + 1e-5)
        norm = np.clip(norm, 0.0, self.reward_clip)
        return norm


class RNDVecEnvWrapper(VecEnvWrapper):
    """Adds scaled RND intrinsic reward to the extrinsic reward on every step."""

    def __init__(
        self,
        venv: VecEnv,
        rnd_module: RNDModule,
        *,
        intrinsic_coef: float = 0.5,
    ) -> None:
        super().__init__(venv)
        self.rnd = rnd_module
        self.intrinsic_coef = intrinsic_coef

    def reset(self) -> np.ndarray:
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        intrinsic = self.rnd.compute_and_train(obs)
        rewards = rewards + self.intrinsic_coef * intrinsic
        for i, info in enumerate(infos):
            info["intrinsic_reward"] = float(intrinsic[i])
        return obs, rewards, dones, infos
