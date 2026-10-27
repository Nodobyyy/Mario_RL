"""Diagnostic seed search, equivalent to upstream best-of-N video selection."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.training.compat import load_runtime, model_custom_objects
from backend.app.storage import UPSTREAM
from collections import deque
import random
import numpy as np
import torch
from stable_baselines3 import PPO
rt=load_runtime()
torch.set_num_threads(2)
path=UPSTREAM/'pretrained/ppo_mario_1-1/best_eval.zip'
model=PPO.load(path,device='cpu',custom_objects=model_custom_objects(path))
cfg=rt.load_env_config_for_model(path)
cfg.n_envs,cfg.noop_max,cfg.end_on_flag=1,0,True
env=rt.make_single_env(cfg)
try:
    for seed in [2,60,42,0,1,3,4,5,6,7,8,9,10,11,12,13,14,15]:
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        obs,_=env.reset(seed=seed)
        stack=deque([obs]*cfg.frame_stack,maxlen=cfg.frame_stack)
        farthest=0
        for step in range(4000):
            action,_=model.predict(rt.stack_observations(stack),deterministic=False)
            obs,_,term,trunc,info=env.step(int(action));stack.append(obs)
            farthest=max(farthest,int(info.get('x_pos',0)))
            if term or trunc:break
        print('SEED',seed,'FLAG',info.get('flag_get',False),'X',farthest,'STEPS',step+1,flush=True)
        if info.get('flag_get'):break
finally:
    env.close()
