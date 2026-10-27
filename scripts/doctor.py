import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.hardware import hardware
from backend.app.storage import UPSTREAM
from backend.training.compat import load_runtime, model_custom_objects, load_vecnormalize


def main():
    status = hardware()
    print(json.dumps(status, indent=2, ensure_ascii=False))
    if status['missing'] or status['torch_error']:
        raise RuntimeError('Missing dependencies; run setup first.')
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(2)
    rt = load_runtime()
    model_path = UPSTREAM / 'pretrained/ppo_mario_1-1/best_eval.zip'
    cfg = rt.load_env_config_for_model(model_path)
    cfg.n_envs, cfg.noop_max = 1, 0
    env = rt.make_vec_env(cfg)
    try:
        obs = env.reset()
        model = PPO.load(model_path, device='cpu', custom_objects=model_custom_objects(model_path))
        action, _ = model.predict(obs, deterministic=True)
        _, reward, _, info = env.step(action)
        frame = rt.render_rgb_frame(env.envs[0])
        assert frame.shape == (240, 256, 3), frame.shape
        normalizer = load_vecnormalize(model_path.parent / 'vecnormalize.pkl', env)
        assert normalizer.ret_rms.count > 0
        print('VECNORMALIZE OK: pretrained reward RMS count=', normalizer.ret_rms.count)
        print('DOCTOR OK: real NES step, RGB frame, pretrained CPU inference.', reward, info[0].get('x_pos'))
    finally:
        env.close()


if __name__ == '__main__':
    main()
