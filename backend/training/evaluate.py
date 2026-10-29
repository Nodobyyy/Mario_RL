from collections import deque
from pathlib import Path
import time
import numpy as np
import torch
from backend.app import storage as s
from backend.app.hardware import resolve_device
from .compat import load_runtime, model_custom_objects
from .frames import FramePublisher
from .telemetry import Telemetry
from .algorithms import algorithm_class, checkpoint_algorithm


def evaluate(job_id):
    job = s.get_job(job_id)
    cfg = job['config']
    cp = s.checkpoint(cfg['checkpoint_id'])
    rt = load_runtime()
    torch.set_num_threads(cfg['threads'])
    torch.manual_seed(cfg['seed'])
    algorithm = checkpoint_algorithm(cp['path'])
    device = resolve_device(cfg['device'])
    model = algorithm_class(algorithm).load(cp['path'], device=device, custom_objects=model_custom_objects(cp['path']))
    env_cfg = rt.load_env_config_for_model(cp['path'])
    env_cfg.world = cfg.get('world') or env_cfg.world
    env_cfg.stage = cfg.get('stage') or env_cfg.stage
    env_cfg.goal_line_x = 0
    env_cfg.n_envs, env_cfg.noop_max, env_cfg.end_on_flag = 1, 0, True
    publisher = FramePublisher(job_id, 'independent_evaluation', cfg['fps'])
    env = Telemetry(rt.make_single_env(env_cfg), job_id, 0, rt, publisher,
                    kind='evaluation', checkpoint=cfg['checkpoint_id'])
    writer = None
    flags, returns, lengths, maxima = [], [], [], []
    started = time.monotonic()
    steps = 0
    stopped = False
    s.patch_job(job_id, status='running', state={'checkpoint_timesteps': int(model.num_timesteps), 'mode': 'independent_evaluation',
        'algorithm': algorithm, 'world': env_cfg.world, 'stage': env_cfg.stage, 'device': device,
        'device_fallback': cfg['device'] == 'cuda' and device == 'cpu'})
    try:
        if cfg['record_video']:
            import imageio.v2 as imageio
            writer = imageio.get_writer(s.job_dir(job_id) / 'evaluation.mp4', fps=cfg['fps'],
                                         codec='libx264', pixelformat='yuv420p', macro_block_size=1)
        for index in range(cfg['episodes']):
            import random
            random.seed(cfg['seed'] + index)
            np.random.seed(cfg['seed'] + index)
            torch.manual_seed(cfg['seed'] + index)
            obs, _ = env.reset(seed=cfg['seed'] + index)
            stack = deque([obs] * env_cfg.frame_stack, maxlen=env_cfg.frame_stack)
            reward_sum, farthest, length = 0.0, 0, 0
            info = {}
            if writer:
                writer.append_data(rt.render_rgb_frame(env))
            for step in range(cfg['max_steps']):
                tick = time.monotonic()
                if s.get_job(job_id)['command'] == 'stop':
                    stopped = True
                    env.finish('interrupted')
                    break
                action, _ = model.predict(rt.stack_observations(stack), deterministic=cfg['deterministic'])
                obs, reward, terminated, truncated, info = env.step(int(action))
                stack.append(obs)
                reward_sum += float(reward)
                farthest = max(farthest, env.max_x)
                steps += 1
                length = step + 1
                if writer:
                    writer.append_data(rt.render_rgb_frame(env))
                if step % 15 == 0 or terminated or truncated:
                    s.patch_job(job_id, state={'eval_steps': steps, 'current': env.current,
                                               'episodes': len(flags), 'elapsed_seconds': time.monotonic() - started})
                if cfg['realtime']:
                    time.sleep(max(0, 1 / cfg['fps'] - (time.monotonic() - tick)))
                if terminated or truncated:
                    break
            if stopped:
                break
            # Simulator may not terminate before the evaluation budget.
            env.finish('evaluation_limit')
            flags.append(bool(info.get('flag_get', False)))
            returns.append(reward_sum)
            lengths.append(length)
            maxima.append(farthest)
        summary = rt.build_rollout_summary(episodes=len(flags), deterministic=cfg['deterministic'],
                   episode_returns=returns, episode_lengths=lengths, episode_flags=flags,
                   max_x_positions=maxima, goal_line_x=rt.effective_goal_line_x(env_cfg))
        summary.update(eval_steps=steps, episodes=len(flags), elapsed_seconds=time.monotonic() - started,
                       checkpoint_id=cfg['checkpoint_id'], checkpoint_timesteps=int(model.num_timesteps),
                       video_available=bool(writer and steps))
        (s.job_dir(job_id) / 'summary.json').write_text(__import__('json').dumps(summary, indent=2), encoding='utf-8')
        if writer:
            writer.close()
            writer = None
        s.metric(job_id, summary)
        s.patch_job(job_id, status='stopped' if stopped else 'completed', state=summary)
        return summary
    finally:
        if writer:
            writer.close()
        env.close()
        publisher.close()
