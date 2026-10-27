"""Adapt upstream main() through factories/callbacks, without rewriting PPO."""
import json
import sys
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage
from backend.app import storage as s
from backend.app.hardware import resolve_device
from .compat import load_runtime, model_custom_objects, load_vecnormalize
from .frames import FramePublisher
from .telemetry import Telemetry
from .callbacks import DashboardCallback


def make_worker_env(env_config, seed, job_id, worker_id, fps):
    rt = load_runtime()
    if isinstance(env_config, dict):
        env_config = rt.config_from_dict(env_config)
    publisher = FramePublisher(job_id, 'training_preview', fps) if worker_id == 0 else None
    env = Telemetry(rt.make_single_env(env_config, seed=seed), job_id, worker_id, rt, publisher)
    original_close = env.close
    def close():
        try:
            original_close()
        finally:
            if publisher:
                publisher.close()
    env.close = close
    return env


def train(job_id):
    cfg = s.get_job(job_id)['config']
    rt = load_runtime()
    import train_mario as upstream
    upstream.VecNormalize.load = staticmethod(load_vecnormalize)
    torch.set_num_threads(cfg['threads'])
    # Upstream does not expose a PPO seed argument; wrap construction to seed it.
    original_ppo = upstream.PPO
    class SeededPPO(original_ppo):
        def __init__(self, *a, **kw):
            kw.setdefault('seed', cfg['seed'])
            super().__init__(*a, **kw)
        @classmethod
        def load(cls, path, *a, **kw):
            custom = model_custom_objects(path)
            custom.update(kw.pop('custom_objects', {}) or {})
            return super().load(path, *a, custom_objects=custom, **kw)
    upstream.PPO = SeededPPO
    saved_argv = sys.argv
    sys.argv = ['train_mario.py']
    args = upstream.parse_args()
    sys.argv = saved_argv
    for name in ('n_envs', 'n_steps', 'batch_size', 'n_epochs', 'learning_rate', 'gamma', 'ent_coef',
                 'checkpoint_freq', 'eval_freq', 'eval_episodes', 'eval_steps', 'seed', 'vec_backend'):
        setattr(args, name, cfg[name])
    args.device = resolve_device(cfg['device'])
    args.timesteps = cfg['total_timesteps']  # additional steps when resuming
    args.run_dir = str(s.job_dir(job_id))
    args.preview_freq = 0
    args.normalize_reward = True
    args.rnd_coef = 0.0  # PPO baseline; upstream RND has no restorable state format.
    args.action_set = 'simple'
    args.backward_penalty_scale = 0.05
    args.flag_bonus = 200.0
    args.stall_steps = 120
    args.stall_penalty = 0.0
    args.milestone_step, args.milestone_bonus = 50, 0.5
    args.hurdle_x, args.hurdle_bonus = [600, 1200, 1800, 2400, 3000], 10.0
    args.action_bias_jump = 1.0
    if cfg.get('resume_checkpoint'):
        cp = s.checkpoint(cfg['resume_checkpoint'])
        args.resume_model = cp['path']
        env_cfg = rt.load_env_config_for_model(cp['path'])
        for key, value in rt.config_to_dict(env_cfg).items():
            if key not in ('n_envs', 'vec_backend') and hasattr(args, key):
                setattr(args, key, value)
    upstream.parse_args = lambda: args
    created = []
    from functools import partial
    def make_vec(config, seed=42):
        # Only primitives cross Windows spawn. Deserializing upstream EnvConfig
        # would import mario_runtime before the worker installs its compatibility bridge.
        env_values = rt.config_to_dict(config)
        factories = [partial(make_worker_env, env_values, seed + i, job_id, i, cfg['preview_fps'])
                     for i in range(config.n_envs)]
        base = SubprocVecEnv(factories, start_method='spawn') if config.vec_backend == 'subproc' and config.n_envs > 1 else DummyVecEnv(factories)
        env = VecTransposeImage(VecFrameStack(base, n_stack=config.frame_stack))
        created.append(env)
        return env
    upstream.make_vec_env = make_vec
    callbacks = []
    def build_callbacks(**kwargs):
        cb = DashboardCallback(job_id, cfg)
        callbacks.append(cb)
        return [cb]
    upstream.build_callbacks = build_callbacks
    s.patch_job(job_id, state={'requested_device': cfg['device'], 'device': args.device,
                              'device_fallback': cfg['device'] == 'cuda' and args.device == 'cpu'})
    try:
        upstream.main()
        cb = callbacks[-1]
        status = 'stopped' if cb.stop_requested else 'completed'
        s.patch_job(job_id, status=status)
    except BaseException:
        for cb in reversed(callbacks):
            if cb.initialized:
                try:
                    cb.save_checkpoint('emergency')
                except Exception:
                    import traceback
                    traceback.print_exc()
                break
        raise
    finally:
        for env in created:
            try:
                env.close()
            except Exception:
                pass
