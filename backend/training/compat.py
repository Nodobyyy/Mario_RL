"""Compatibility bridge; upstream files remain byte-for-byte unchanged."""
import sys
import types
from functools import lru_cache


@lru_cache(maxsize=1)
def load_runtime():
    from backend.app.storage import UPSTREAM
    sys.path.insert(0, str(UPSTREAM))
    import gymnasium as gym
    import gym_super_mario_bros.actions as actions
    import gym_super_mario_bros.smb_env as smb
    import numpy as np
    # The shipped policy was pickled with NumPy 2, while legacy NES needs 1.x.
    # These are module-path aliases to identical public NumPy 1.x objects.
    import numpy.core.numeric
    import numpy.core.multiarray
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)

    # Upstream imports an unpublished `gymnasium_super_mario_bros` fork.
    # Alias the official package, bridge spaces, retain its legacy step/reset API
    # for upstream LegacyMarioToGymnasium to convert exactly once.
    class CompatibleMario(smb.SuperMarioBrosEnv, gym.Env):
        observation_space = gym.spaces.Box(0, 255, shape=(240, 256, 3), dtype=np.uint8)
        action_space = gym.spaces.Discrete(256)
        _cursor_init_patch = True
        _cursor_step_patch = True

    alias = types.ModuleType('gymnasium_super_mario_bros')
    alias.__path__ = []
    module = types.ModuleType('gymnasium_super_mario_bros.smb_env')
    module.SuperMarioBrosEnv = CompatibleMario
    module.decode_target = smb.decode_target
    module.rom_path = smb.rom_path
    sys.modules['gymnasium_super_mario_bros'] = alias
    sys.modules['gymnasium_super_mario_bros.actions'] = actions
    sys.modules['gymnasium_super_mario_bros.smb_env'] = module
    import mario_runtime
    return mario_runtime


def model_custom_objects(model_path):
    """Recreate spaces and discard transient emulator observations on load.

    NumPy 2 pickles cannot be round-tripped by NumPy 1.26. PPO weights and Adam
    tensors are stored separately; replacing these transient values preserves
    the policy/optimizer while starting a new emulator episode, as SB3 does.
    """
    import gymnasium as gym
    import numpy as np
    rt = load_runtime()
    cfg = rt.load_env_config_for_model(model_path)
    actions = rt.ACTION_SET_MAP[cfg.action_set]
    return {'_last_obs': None, '_last_original_obs': None, '_last_episode_starts': None,
            'observation_space': gym.spaces.Box(0, 255, (cfg.frame_stack, cfg.screen_size, cfg.screen_size), dtype=np.uint8),
            'action_space': gym.spaces.Discrete(len(actions) + int(cfg.long_jump_action))}


def _numpy_frombuffer(buf, dtype, shape, order, axis_order=None):
    import numpy as np
    array = np.frombuffer(buf, dtype=dtype)
    if order == 'K' and axis_order is not None:
        return array.reshape(shape, order='C').transpose(axis_order)
    return array.reshape(shape, order=order)


class _SpaceRNGPlaceholder:
    """Only used for obsolete space RNG state; actual env spaces replace it."""
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


def load_vecnormalize(path, env):
    """Restore exact running means/variances across NumPy 2 → 1.26.

    Emulation starts a new episode, so the current env owns the action/observation
    spaces and their RNGs. Reward RMS and all normalizer parameters are retained.
    """
    import pickle
    class CompatibleUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module in ('numpy._core.numeric', 'numpy.core.numeric') and name == '_frombuffer':
                return _numpy_frombuffer
            if module == 'numpy.random._pickle' and name in ('__bit_generator_ctor', '__generator_ctor', '__randomstate_ctor'):
                return _SpaceRNGPlaceholder
            return super().find_class(module, name)
    with open(path, 'rb') as f:
        normalizer = CompatibleUnpickler(f).load()
    normalizer.observation_space = env.observation_space
    normalizer.action_space = env.action_space
    normalizer.set_venv(env)
    return normalizer
