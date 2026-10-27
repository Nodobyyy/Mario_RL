import glob, numpy as np
from pathlib import Path
LOGDIR = Path(r"C:\Users\31660\mario-rl\dreamerv3\logs")
for run in ["mario_run4_plan2explore", "mario_run3_rescue", "mario_run1"]:
    eps = sorted(glob.glob(str(LOGDIR / run / "eval_eps" / "*.npz")))
    if not eps:
        continue
    z = np.load(eps[-1])
    print(run, "files=", z.files)
    if "reward" in z.files:
        print("  reward sum=", float(z["reward"].sum()), " len=", len(z["reward"]))
    break
