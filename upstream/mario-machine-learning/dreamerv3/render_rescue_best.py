import numpy as np, glob, os, imageio.v2 as imageio
os.makedirs('viz', exist_ok=True)

# Get all rescue episodes, find the top 3 by forward progress
files = glob.glob('logs/mario_run3_rescue/eval_eps/*.npz') + \
        glob.glob('logs/mario_run3_rescue/train_eps/*.npz')

scored = []
for f in files:
    d = np.load(f)
    rewards = d['reward']
    # forward_bonus only (ignore death and flag)
    fwd = float(sum(r for r in rewards if 0 < r < 50))
    scored.append((f, fwd, len(rewards)))
scored.sort(key=lambda x: -x[1])
print('Top 5 rescue episodes by forward progress:')
for f, fwd, n in scored[:5]:
    src = 'eval' if 'eval_eps' in f else 'train'
    print(f'  [{src}] fwd~{fwd/0.05:.0f}px  len={n}  {os.path.basename(f)}')

for i, (f, fwd, n) in enumerate(scored[:3]):
    d = np.load(f)
    frames = d['image']
    big = np.repeat(np.repeat(frames, 3, axis=1), 3, axis=2)
    src = 'eval' if 'eval_eps' in f else 'train'
    out = f'viz/RESCUE_TOP{i+1}_{src}_fwd{fwd/0.05:.0f}.gif'
    imageio.mimsave(out, big, duration=1/30, loop=0)
    print(f'wrote: {out}')
