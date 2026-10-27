"""Root-cause diagnosis for mario_run4_plan2explore.

Goals:
  1. Did Mario *ever* reach the flag in any run? (max single-step reward >= flag_bonus)
  2. How far did the *best* episode actually go? (proxy via cumulative
     forward_bonus + hurdle_bonus, since x_pos is not stored in npz)
  3. What changed in the per-step reward distribution before vs after the
     R8 (step_penalty + goal_potential) switch?
  4. Are intrinsic rewards from Plan2Explore drowning out extrinsic in the
     loss? (read latest train metrics from log)
"""
import json, glob, os
from pathlib import Path
import numpy as np

LOGDIR = Path(r"C:\Users\31660\mario-rl\dreamerv3\logs")
RUNS = ["mario_run1", "mario_run3_rescue", "mario_run4_plan2explore"]

# Reward shaping constants we configured (per configs.yaml)
FLAG_BONUS = 100.0       # one-time on flag_get
DEATH_PEN  = 50.0        # one-time on life loss
HURDLE     = 5.0         # every 200 px of new max-x
STEP_PEN_R8 = 0.005      # only enabled in run4 after step ~520k


def load_eps(run, kind="eval_eps"):
    return sorted(glob.glob(str(LOGDIR / run / kind / "*.npz")))


def episode_summary(npz_path):
    data = np.load(npz_path)
    r = data["reward"].astype(np.float32)
    is_term = bool(data["is_terminal"][-1]) if "is_terminal" in data.files else False
    return {
        "len": len(r),
        "sum": float(r.sum()),
        "max_step": float(r.max()),
        "min_step": float(r.min()),
        "n_hurdles": int((r >= HURDLE - 0.1).sum()),  # >=5 single steps
        "got_flag": bool((r >= FLAG_BONUS - 0.1).any()),
        "died": bool((r <= -DEATH_PEN + 0.1).any()),
        "is_terminal": is_term,
        "mtime": os.path.getmtime(npz_path),
    }


def fmt(x, w=8, p=2):
    return f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) else f"{x:>{w}}"


print("=" * 78)
print("Q1: Did Mario EVER reach the flag in any run?")
print("=" * 78)
flag_count_per_run = {}
for run in RUNS:
    n = 0
    eps = load_eps(run, "train_eps") + load_eps(run, "eval_eps")
    for ep in eps:
        try:
            s = episode_summary(ep)
            if s["got_flag"]:
                n += 1
        except Exception:
            pass
    flag_count_per_run[run] = (n, len(eps))
    print(f"  {run:30s}  flag_get episodes: {n:4d} / {len(eps):4d}")

print()
print("=" * 78)
print("Q2: Best (highest reward sum) episode per run")
print("=" * 78)
for run in RUNS:
    eps = load_eps(run, "train_eps") + load_eps(run, "eval_eps")
    summaries = []
    for ep in eps:
        try:
            s = episode_summary(ep)
            s["path"] = os.path.basename(ep)
            summaries.append(s)
        except Exception:
            pass
    if not summaries:
        continue
    top = sorted(summaries, key=lambda s: s["sum"], reverse=True)[:3]
    print(f"\n  --- {run} ---")
    print(f"  {'#':>2} {'len':>5} {'sum':>8} {'max':>7} {'min':>7} {'hurdles':>7} {'died':>5} {'flag':>5}  episode")
    for i, s in enumerate(top):
        print(f"  {i+1:>2} {s['len']:>5d} {s['sum']:>8.1f} {s['max_step']:>7.2f} "
              f"{s['min_step']:>7.2f} {s['n_hurdles']:>7d} "
              f"{str(s['died']):>5} {str(s['got_flag']):>5}  {s['path']}")

print()
print("=" * 78)
print("Q3: run4 -- per-step reward profile BEFORE vs AFTER R8 switch")
print("=" * 78)
# R8 switched in around step 511712-520120 (when we relaunched).
# Use file mtime to bucket: episodes whose npz mtime is in the last ~hour
# = "AFTER R8". Anything older = "BEFORE R8".
import datetime as dt
now = dt.datetime.now().timestamp()
CUT = now - 60 * 60  # 1 hour

def bucket(eps, after):
    out = []
    for ep in eps:
        try:
            s = episode_summary(ep)
            if (s["mtime"] >= CUT) == after:
                out.append(s)
        except Exception:
            pass
    return out

run = "mario_run4_plan2explore"
eps = load_eps(run, "eval_eps")
before = bucket(eps, after=False)
after  = bucket(eps, after=True)

def stats(label, group):
    if not group:
        print(f"  {label:<8} (no episodes)")
        return
    lens = [g["len"] for g in group]
    sums = [g["sum"] for g in group]
    deaths = sum(1 for g in group if g["died"])
    hurd = sum(g["n_hurdles"] for g in group) / len(group)
    flags = sum(1 for g in group if g["got_flag"])
    print(f"  {label:<8} n={len(group):3d}  len mean={np.mean(lens):6.0f} median={np.median(lens):6.0f}  "
          f"return mean={np.mean(sums):7.2f}  deaths={deaths}/{len(group)}  "
          f"avg_hurdles/ep={hurd:.2f}  flags={flags}")

stats("BEFORE", before)
stats("AFTER",  after)

print()
print("=" * 78)
print("Q4: Look at the LATEST eval episode in detail (where does Mario die?)")
print("=" * 78)
if eps:
    latest = sorted(eps, key=os.path.getmtime)[-1]
    data = np.load(latest)
    r = data["reward"].astype(np.float32)
    print(f"  episode file : {os.path.basename(latest)}")
    print(f"  length       : {len(r)} env steps")
    print(f"  total reward : {r.sum():.2f}")

    # Find the death step (first step <= -DEATH_PEN+0.1)
    death_idx = np.where(r <= -DEATH_PEN + 0.1)[0]
    if len(death_idx):
        di = int(death_idx[0])
        print(f"  death at step {di} / {len(r)}  (reward={r[di]:.2f})")
        print(f"  reward in 10 steps before death:")
        lo = max(0, di - 10)
        for i in range(lo, di + 1):
            tag = " <- DEATH" if i == di else ""
            print(f"    step {i:>4}  r={r[i]:>8.3f}{tag}")
    else:
        print("  no death detected; episode ended via timeout/flag")

    # Decompose total reward by sign
    pos = r[r > 0].sum()
    neg = r[r < 0].sum()
    print()
    print(f"  positive reward sum : {pos:>8.2f}")
    print(f"  negative reward sum : {neg:>8.2f}")
    print(f"  step_penalty if active (-0.005 * len) = {-STEP_PEN_R8 * len(r):.2f}")

print()
print("=" * 78)
print("Q5: Latest training-side numbers (extrinsic vs intrinsic gradient signal)")
print("=" * 78)
mfile = LOGDIR / run / "metrics.jsonl"
rows = [json.loads(l) for l in mfile.read_text(encoding="utf-8").splitlines() if l.strip()]
last_train = next((r for r in reversed(rows) if "actor_loss" in r), {})
keys = [
    "step",
    "actor_loss", "value_loss", "actor_entropy",
    "imag_reward_mean", "imag_reward_min", "imag_reward_max",
    "expl_imag_reward_mean", "expl_imag_reward_min", "expl_imag_reward_max",
    "EMA_005", "EMA_095", "expl_EMA_005", "expl_EMA_095",
]
for k in keys:
    if k in last_train:
        v = last_train[k]
        try: print(f"  {k:<24} = {float(v):>9.3f}")
        except Exception: print(f"  {k:<24} = {v}")
