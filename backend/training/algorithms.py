"""Supported training algorithms and checkpoint dispatch."""
import json
from pathlib import Path
from zipfile import ZipFile


def algorithm_class(name):
    from stable_baselines3 import PPO, A2C
    return {'ppo': PPO, 'a2c': A2C}[name]


def checkpoint_algorithm(path):
    with ZipFile(path) as archive:
        data = json.loads(archive.read('data'))
    return data.get('dashboard_algorithm', 'ppo')


def checkpoint_details(path):
    result = {'algorithm': checkpoint_algorithm(path), 'world': 1, 'stage': 1}
    for parent in Path(path).parents:
        config = parent / 'train_config.json'
        if config.is_file():
            env = json.loads(config.read_text(encoding='utf-8')).get('env', {})
            result.update(world=env.get('world', 1), stage=env.get('stage', 1))
            break
    return result
