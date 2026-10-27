"""Real progress tracker using the no-oscillation reward.

Under the new reward, forward_bonus = 0.05 * delta_new_max (only triggers on new
max-x). So for each episode, sum of positive rewards = 0.05 * final_max_x.

This gives us the TRUE max-x reached, not oscillation-inflated numbers.
"""
import numpy as np, glob, os, sys, json
from datetime import datetime

LOGDIR = 'logs/mario_run3_rescue'

def ep_stats(path):
    d = np.load(path)
    r = d['reward']
    # Episode ended short (<1900 steps) = died early; length 2005-2006 = NES timer kill
    # Any episode ending without flag is a "death" under single-life mode, but we
    # distinguish "early combat/pit death" vs "ran out of NES in-game timer".
    ep_len = len(r)
    cleared = bool(r.max() > 50)  # flag_bonus is +100 which exceeds 50
    early_death = (not cleared) and (ep_len < 1900)
    timer_death = (not cleared) and (ep_len >= 1900)
    # Sum of positive step rewards (excluding flag bonus) = 0.05 * max_x under
    # no-oscillation reward.
    pos_sum = float(sum(x for x in r if 0 < x < 50))
    max_x = pos_sum / 0.05
    return dict(len=ep_len, max_x=max_x, early_death=early_death,
                timer_death=timer_death, cleared=cleared,
                mtime=os.path.getmtime(path))

def main():
    eval_files = sorted(glob.glob(f'{LOGDIR}/eval_eps/*.npz'), key=os.path.getmtime)
    train_files = sorted(glob.glob(f'{LOGDIR}/train_eps/*.npz'), key=os.path.getmtime)

    def tag(s):
        if s['cleared']:
            return 'FLAG!'
        if s['early_death']:
            return 'DIED'
        if s['timer_death']:
            return 'TIMER'
        return 'TIMEOUT'

    print(f'\n=== LAST 10 EVAL episodes (real max-x) ===')
    for f in eval_files[-10:]:
        s = ep_stats(f)
        when = datetime.fromtimestamp(s['mtime']).strftime('%H:%M:%S')
        print(f'  {when}  max_x={s["max_x"]:6.0f}  len={s["len"]:4d}  [{tag(s)}]')

    print(f'\n=== LAST 10 TRAIN episodes ===')
    for f in train_files[-10:]:
        s = ep_stats(f)
        when = datetime.fromtimestamp(s['mtime']).strftime('%H:%M:%S')
        print(f'  {when}  max_x={s["max_x"]:6.0f}  len={s["len"]:4d}  [{tag(s)}]')

    # Rolling max_x average (last 20)
    all_files = sorted(eval_files + train_files, key=os.path.getmtime)
    recent = [ep_stats(f) for f in all_files[-20:]]
    if recent:
        avg = np.mean([s['max_x'] for s in recent])
        best = max(s['max_x'] for s in recent)
        clears = sum(1 for s in recent if s['cleared'])
        print(f'\n=== ROLLING STATS (last {len(recent)} episodes) ===')
        print(f'  avg max_x: {avg:.0f}  |  best max_x: {best:.0f}  |  clears: {clears}')

    # Best ever
    all_stats = [ep_stats(f) for f in all_files]
    if all_stats:
        best = max(all_stats, key=lambda s: s['max_x'])
        print(f'\n=== BEST EVER (all {len(all_stats)} rescue episodes) ===')
        print(f'  max_x={best["max_x"]:.0f}  len={best["len"]}  cleared={best["cleared"]}')

if __name__ == '__main__':
    main()
