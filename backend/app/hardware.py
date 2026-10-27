import importlib.util
import os
import platform
import sys
from functools import lru_cache
from .storage import UPSTREAM


@lru_cache(maxsize=1)
def hardware():
    cuda, gpu, vram = False, None, None
    torch_version, torch_error = None, None
    try:
        import torch
        torch_version = torch.__version__
        cuda = torch.cuda.is_available()
        if cuda:
            gpu = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    except Exception as e:
        torch_error = str(e)
    missing = [x for x in ('torch', 'stable_baselines3', 'gym_super_mario_bros', 'nes_py', 'cv2')
               if importlib.util.find_spec(x) is None]
    return dict(python=sys.version.split()[0], platform=platform.system(),
                cpu=platform.processor(), cpu_count=os.cpu_count() or 1,
                cuda=cuda, gpu=gpu, vram_gb=vram, torch=torch_version,
                missing=missing, torch_error=torch_error,
                upstream_ready=(UPSTREAM / 'train_mario.py').exists(),
                profile='gpu' if cuda else 'cpu',
                defaults=dict(device='auto', n_envs=8 if cuda else 1,
                              n_steps=512 if cuda else 128, batch_size=256 if cuda else 64,
                              n_epochs=4, threads=min(4, os.cpu_count() or 1)))


def resolve_device(requested):
    import torch
    # Explicit cuda gracefully falls back if unavailable, as does auto.
    return 'cuda' if requested != 'cpu' and torch.cuda.is_available() else 'cpu'
