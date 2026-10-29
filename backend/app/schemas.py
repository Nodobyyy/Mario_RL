from typing import Literal
from pydantic import BaseModel, Field, model_validator


class RunConfig(BaseModel):
    name: str = Field('PPO · Mario 1-1', min_length=1, max_length=100)
    algorithm: Literal['ppo', 'a2c'] = 'ppo'
    world: int = Field(1, ge=1, le=8)
    stage: int = Field(1, ge=1, le=4)
    total_timesteps: int = Field(20000, ge=32, le=1_000_000_000)
    device: Literal['auto', 'cpu', 'cuda'] = 'auto'
    n_envs: Literal[1, 4, 8] = 1
    n_steps: int = Field(128, ge=32, le=2048)
    batch_size: int = Field(64, ge=8, le=4096)
    n_epochs: int = Field(4, ge=1, le=20)
    learning_rate: float = Field(0.00025, gt=0, le=0.01)
    gamma: float = Field(0.9, gt=0, le=1)
    ent_coef: float = Field(0.05, ge=0, le=1)
    checkpoint_freq: int = Field(5000, ge=32)
    eval_freq: int = Field(25000, ge=0)
    eval_episodes: int = Field(3, ge=1, le=20)
    eval_steps: int = Field(4000, ge=32, le=20000)
    seed: int = Field(42, ge=0, le=2**31-1)
    threads: int = Field(4, ge=1, le=32)
    preview_fps: int = Field(30, ge=0, le=60)
    vec_backend: Literal['dummy', 'subproc'] = 'dummy'
    resume_checkpoint: str | None = None

    @model_validator(mode='after')
    def check_batch(self):
        if self.algorithm == 'ppo' and (self.batch_size > self.n_envs * self.n_steps or (self.n_envs * self.n_steps) % self.batch_size):
            raise ValueError('batch_size 必须整除 n_envs × n_steps')
        return self


class PreviewConfig(BaseModel):
    fps: int = Field(30, ge=0, le=60)


class EvalConfig(BaseModel):
    checkpoint_id: str = 'pretrained'
    world: int | None = Field(None, ge=1, le=8)
    stage: int | None = Field(None, ge=1, le=4)
    episodes: int = Field(1, ge=1, le=20)
    max_steps: int = Field(4000, ge=32, le=20000)
    deterministic: bool = False
    seed: int = Field(42, ge=0, le=2**31-1)
    fps: int = Field(15, ge=5, le=30)
    realtime: bool = True
    record_video: bool = True
    device: Literal['auto', 'cpu', 'cuda'] = 'cpu'
    threads: int = Field(2, ge=1, le=32)
