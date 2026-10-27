"""DreamerV3-compatible wrapper around gymnasium-super-mario-bros.

Bridges gymnasium-based gym-super-mario-bros into the legacy gym API used by
the rest of the dreamerv3-torch codebase, and produces observations in the
{'image', 'is_first', 'is_terminal'} dict format Dreamer expects.
"""

from __future__ import annotations

import gym
import numpy as np

# Reuse the patches and JoypadSpace mapping we already maintain for the SB3 PPO baseline.
import sys, pathlib
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mario_runtime import patch_super_mario_compat, ACTION_SET_MAP, DEFAULT_ACTIONS
from mario_runtime import LEVEL_1_1_GOAL_X
from gymnasium_super_mario_bros.smb_env import SuperMarioBrosEnv

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


_BUTTON_BIT = {
    "right": 0b10000000,
    "left": 0b01000000,
    "down": 0b00100000,
    "up": 0b00010000,
    "start": 0b00001000,
    "select": 0b00000100,
    "B": 0b00000010,
    "A": 0b00000001,
    "NOOP": 0b00000000,
}


def _action_to_byte(buttons: list[str]) -> int:
    byte = 0
    for b in buttons:
        byte |= _BUTTON_BIT.get(b, 0)
    return byte


class Mario:
    """Single-environment wrapper Dreamer can consume.

    Args:
        task: Stage selector formatted as 'world-stage' (e.g. '1-1').
        action_repeat: Frame-skip; identical action is held for this many raw frames.
        size: Resized observation (H, W) returned to Dreamer.
        gray: If True, return single-channel grayscale image (Dreamer default for Atari).
        time_limit: Maximum env-frames per episode (raw, before action_repeat).
        seed: Optional seed for action / reset randomness.
        action_set: 'right_only' or 'simple'. Determines discrete action count.
        sticky_jump_frames: Optional. If set >0, when action contains 'A' the wrapper
            keeps holding 'A' for this many extra raw frames at the end of an action_repeat
            block. Useful for variable-height jumps without dedicated macros.
    """

    metadata = {}

    def __init__(
        self,
        task: str = "1-1",
        action_repeat: int = 4,
        size=(64, 64),
        gray: bool = False,
        time_limit: int = 4500,
        seed: int | None = None,
        action_set: str = "simple",
        sticky_jump_frames: int = 0,
        flag_base_bonus: float = 100.0,
        time_bonus_scale: float = 0.0,
        score_scale: float = 0.0,
        death_penalty: float = 0.0,
        hurdle_bonus: float = 0.0,
        hurdle_spacing: int = 400,
        step_penalty: float = 0.0,
        goal_potential_scale: float = 0.0,
        goal_potential_gamma: float = 1.0,
        goal_x: int | None = None,
    ):
        patch_super_mario_compat()
        world_str, stage_str = task.split("-")
        self._world = int(world_str)
        self._stage = int(stage_str)
        self._action_repeat = int(action_repeat)
        self._size = tuple(int(x) for x in size)
        self._gray = bool(gray)
        self._time_limit = int(time_limit)
        self._sticky_jump_frames = int(sticky_jump_frames)
        self._flag_base_bonus = float(flag_base_bonus)
        self._time_bonus_scale = float(time_bonus_scale)
        self._score_scale = float(score_scale)
        self._death_penalty = float(death_penalty)
        self._hurdle_bonus = float(hurdle_bonus)
        self._hurdle_spacing = max(1, int(hurdle_spacing))
        self._step_penalty = float(step_penalty)
        self._goal_potential_scale = float(goal_potential_scale)
        self._goal_potential_gamma = float(goal_potential_gamma)
        self._goal_x = int(goal_x) if goal_x is not None else int(LEVEL_1_1_GOAL_X)
        self._crossed_hurdles: set[int] = set()
        self._last_score = 0
        self._random = np.random.RandomState(seed)

        self._env = SuperMarioBrosEnv(target=(self._world, self._stage))
        action_button_lists = ACTION_SET_MAP.get(action_set, DEFAULT_ACTIONS)
        self._action_bytes = [_action_to_byte(buttons) for buttons in action_button_lists]
        self._action_buttons = list(action_button_lists)

        self._step = 0
        self._done = True
        self._last_x = 0
        self._max_x = 0
        self._last_life = None
        self._last_goal_potential = -float(max(0, self._goal_x))

        # Pre-allocate observation buffers.
        raw_shape = self._env.observation_space.shape  # (240, 256, 3)
        self._obs_buffer = np.zeros((2, *raw_shape), dtype=np.uint8)

    # ------------------------------------------------------------------
    # Spaces (use legacy gym so Dreamer's wrappers Just Work).
    # ------------------------------------------------------------------
    @property
    def observation_space(self):
        channels = 1 if self._gray else 3
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    0, 255, (*self._size, channels), dtype=np.uint8
                ),
            }
        )

    @property
    def action_space(self):
        space = gym.spaces.Discrete(len(self._action_bytes))
        space.discrete = True
        return space

    # ------------------------------------------------------------------
    # Core API.
    # ------------------------------------------------------------------
    def reset(self):
        raw = self._env.reset()
        if isinstance(raw, tuple):
            raw = raw[0]
        self._obs_buffer[0] = raw
        self._obs_buffer[1].fill(0)
        self._step = 0
        self._done = False
        self._last_x = 0
        self._max_x = 0
        self._last_score = 0
        self._last_life = None
        self._crossed_hurdles = set()
        self._last_goal_potential = -float(max(0, self._goal_x - self._last_x))
        obs, _, _, _ = self._obs(0.0, is_first=True)
        return obs

    def step(self, action):
        if hasattr(action, "shape") and getattr(action, "shape", ()):
            action_idx = int(np.argmax(action))
        else:
            action_idx = int(action)
        action_byte = self._action_bytes[action_idx]
        buttons = self._action_buttons[action_idx]
        is_jump = "A" in buttons

        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict = {}

        held_frames = self._action_repeat
        if is_jump and self._sticky_jump_frames > 0:
            held_frames += self._sticky_jump_frames

        # Treat single-life losses as episode termination so the agent gets a
        # prompt death signal instead of NES respawning silently. Without this,
        # Mario quietly loses a life and continues, and our death_penalty never
        # triggers until all 3 lives are gone (which in practice doesn't happen
        # within our time budget).
        life_lost = False
        for repeat in range(held_frames):
            obs_raw, reward, terminated, info = self._env.step(action_byte)
            self._step += 1
            total_reward += float(reward)
            if repeat == held_frames - 2:
                self._obs_buffer[1] = obs_raw
            if self._last_life is None:
                self._last_life = info.get("life", None)
            else:
                cur_life = info.get("life", None)
                if cur_life is not None and cur_life < self._last_life:
                    life_lost = True
                self._last_life = cur_life
            if terminated or life_lost:
                break
            if self._time_limit and self._step >= self._time_limit:
                truncated = True
                break
        # Make sure final frame is always captured for max-pooling.
        self._obs_buffer[0] = obs_raw

        # Reward is granted ONLY when Mario reaches a NEW max-x within this
        # episode. This prevents the "oscillation exploit" where an agent
        # repeatedly walks forward-backward-forward at a choke point and
        # accumulates positive reward without actually making progress. Under
        # this scheme, oscillation nets 0 reward so the only way to gain
        # reward is to push past the previous best x-position.
        x_now = int(info.get("x_pos", self._last_x))
        self._last_x = x_now
        delta_new_max = max(0, x_now - self._max_x)
        self._max_x = max(self._max_x, x_now)

        forward_bonus = 0.05 * delta_new_max

        # Hurdle bonus: one-time reward for crossing fixed x thresholds within
        # the episode. Densifies the forward signal without changing the max-x
        # invariant (still rewards real progress, not oscillation, since each
        # hurdle is logged and only paid out once per episode).
        hurdle_reward = 0.0
        if self._hurdle_bonus > 0 and delta_new_max > 0:
            # Find all hurdles newly crossed by this max-x update.
            low = (self._max_x - delta_new_max) // self._hurdle_spacing
            high = self._max_x // self._hurdle_spacing
            for h in range(int(low) + 1, int(high) + 1):
                if h not in self._crossed_hurdles:
                    self._crossed_hurdles.add(h)
                    hurdle_reward += self._hurdle_bonus

        flag_get = bool(info.get("flag_get", False))
        flag_bonus = self._flag_base_bonus if flag_get else 0.0

        # Speed bonus: only on successful flag_get. info['time'] is NES in-game timer
        # that starts at 400 and counts down. Faster clear -> more remaining time -> larger bonus.
        # Always non-negative; never subtracted when agent fails to clear.
        time_remaining = int(info.get("time", 0) or 0)
        time_bonus = self._time_bonus_scale * time_remaining if flag_get else 0.0

        # Per-step score delta (coins, enemy kills, etc.). Scaled small by default.
        score_now = int(info.get("score", 0) or 0)
        delta_score = max(0, score_now - self._last_score)
        self._last_score = score_now
        score_bonus = self._score_scale * delta_score

        # Optional one-time death penalty. Treat both NES game-over and any
        # single-life loss as a death event (we end the episode on life loss).
        is_terminal = bool((terminated or life_lost) and not flag_get)
        death_pen = -self._death_penalty if is_terminal and self._death_penalty > 0 else 0.0

        # Small per-decision time cost to discourage "safe but slow" policies.
        step_pen = -self._step_penalty if self._step_penalty > 0 else 0.0

        # Potential-based shaping toward the stage goal:
        #   r = scale * (gamma * phi(s') - phi(s)), phi(s) = -distance_to_goal.
        # This rewards reducing remaining distance while preserving the optimal policy.
        potential_reward = 0.0
        if self._goal_potential_scale > 0:
            distance_now = max(0, self._goal_x - x_now)
            goal_potential_now = -float(distance_now)
            potential_reward = self._goal_potential_scale * (
                self._goal_potential_gamma * goal_potential_now - self._last_goal_potential
            )
            self._last_goal_potential = goal_potential_now

        shaped = (
            forward_bonus
            + hurdle_reward
            + flag_bonus
            + time_bonus
            + score_bonus
            + death_pen
            + step_pen
            + potential_reward
        )

        is_last = bool(terminated or truncated or flag_get or life_lost)
        self._done = is_last
        return self._obs(shaped, is_last=is_last, is_terminal=is_terminal)

    # ------------------------------------------------------------------
    # Observation post-processing.
    # ------------------------------------------------------------------
    def _obs(self, reward: float, is_first: bool = False, is_last: bool = False, is_terminal: bool = False):
        np.maximum(self._obs_buffer[0], self._obs_buffer[1], out=self._obs_buffer[0])
        image = self._obs_buffer[0]
        if image.shape[:2] != self._size:
            if cv2 is not None:
                image = cv2.resize(image, (self._size[1], self._size[0]), interpolation=cv2.INTER_AREA)
            else:
                from PIL import Image  # type: ignore
                image = np.array(Image.fromarray(image).resize((self._size[1], self._size[0]), Image.NEAREST))
        if self._gray:
            weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
            image = np.tensordot(image, weights, (-1, 0)).astype(np.uint8)[:, :, None]
        obs_dict = {
            "image": image,
            "is_first": is_first,
            "is_terminal": is_terminal,
        }
        return obs_dict, reward, is_last, {"max_x": self._max_x, "x_pos": self._last_x}

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass
