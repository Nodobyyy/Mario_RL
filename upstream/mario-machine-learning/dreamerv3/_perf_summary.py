import json, os, glob, sys
from pathlib import Path

LOGDIR = Path(r"C:\Users\31660\mario-rl\dreamerv3\logs")

runs = ["mario_run1", "mario_run3_rescue", "mario_run4_plan2explore"]

for run in runs:
    p = LOGDIR / run / "metrics.jsonl"
    if not p.exists():
        print(f"-- {run}: no metrics --")
        continue
    lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        print(f"-- {run}: empty --")
        continue
    rows = [json.loads(l) for l in lines]
    last = rows[-1]
    eval_rows = [r for r in rows if "eval_return" in r]
    best_eval = max((r["eval_return"] for r in eval_rows), default=None)
    last_eval = eval_rows[-1] if eval_rows else None
    train_eps = sorted(glob.glob(str(LOGDIR / run / "train_eps" / "*.npz")))
    eval_eps = sorted(glob.glob(str(LOGDIR / run / "eval_eps" / "*.npz")))

    print(f"=== {run} ===")
    print(f"  total_lines        : {len(lines)}")
    print(f"  latest_step        : {last.get('step')}")
    print(f"  num_train_episodes : {len(train_eps)}")
    print(f"  num_eval_episodes  : {len(eval_eps)}")
    print(f"  best_eval_return   : {best_eval}")
    if last_eval:
        print(f"  latest_eval_return : {last_eval.get('eval_return')} @ step {last_eval.get('step')}")

    import numpy as np
    max_x_overall = 0
    max_x_run = None
    for ep in train_eps + eval_eps:
        try:
            data = np.load(ep)
            if "x_pos" in data.files:
                xs = data["x_pos"]
                m = int(xs.max())
                if m > max_x_overall:
                    max_x_overall = m
                    max_x_run = os.path.basename(ep)
        except Exception:
            pass
    print(f"  best_x_pos_seen    : {max_x_overall}  ({max_x_run})")
    print()
