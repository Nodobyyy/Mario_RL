"""Print a compact table of eval summaries for a run directory."""
import json
import sys
from pathlib import Path


def main() -> None:
    run_dir = Path(sys.argv[1])
    evals = sorted((run_dir / "evaluations").glob("step_*/summary.json"))
    print(f"{'step':>8} {'median_x':>9} {'best_x':>7} {'clear':>6} {'avg_ret':>10} {'avg_len':>8}")
    for p in evals:
        data = json.loads(p.read_text())
        print(
            f"{data['num_timesteps']:>8} "
            f"{data['median_max_x']:>9.1f} "
            f"{data['best_max_x']:>7} "
            f"{data['clear_rate']:>6.2f} "
            f"{data['average_return']:>10.1f} "
            f"{data['average_length']:>8.1f}"
        )


if __name__ == "__main__":
    main()
