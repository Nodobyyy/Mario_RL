from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from stable_baselines3 import PPO

LEVEL_1_1_GOAL_X = 3161


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-refresh the Mario training canvas.")
    parser.add_argument("--terminal-file", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--canvas-file", required=True)
    parser.add_argument("--interval", type=int, default=15)
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_header_command(log_text: str) -> str:
    match = re.search(r'^command: "(.*)"$', log_text, re.MULTILINE)
    if not match:
        return ""
    return match.group(1).replace('\\"', '"')


def parse_arg(command: str, flag: str) -> str:
    pattern = re.compile(rf'{re.escape(flag)}\s+"([^"]+)"|{re.escape(flag)}\s+([^\s"]+)')
    match = pattern.search(command)
    if not match:
        return ""
    return match.group(1) or match.group(2) or ""


def parse_last_metric(log_text: str, label: str) -> float | None:
    matches = re.findall(rf"\|\s+{re.escape(label)}\s+\|\s+([^\|]+?)\s+\|", log_text)
    if not matches:
        return None
    value = matches[-1].strip()
    try:
        return float(value)
    except ValueError:
        return None


def parse_last_int_metric(log_text: str, label: str) -> int | None:
    value = parse_last_metric(log_text, label)
    return int(value) if value is not None else None


def parse_run_status(log_text: str) -> str:
    if "exit_code:" in log_text:
        return "Completed"
    return "Training"


def load_json(path: Path) -> dict:
    return json.loads(read_text(path))


def resolve_active_run_dir(log_text: str, requested_run_dir: Path) -> Path:
    if (requested_run_dir / "train_config.json").exists():
        return requested_run_dir

    logging_matches = re.findall(r"Logging to ([^\r\n]+)", log_text)
    for raw_path in reversed(logging_matches):
        path = Path(raw_path.strip())
        if len(path.parents) >= 3:
            candidate = path.parents[2]
            if (candidate / "train_config.json").exists():
                return candidate

    preview_matches = re.findall(r"video=([^\r\n]+preview\.mp4)", log_text)
    for raw_path in reversed(preview_matches):
        candidate = Path(raw_path.strip()).parent
        if (candidate.parent / "train_config.json").exists():
            return candidate.parent

    phase_matches = re.findall(r"train_ok run_dir=([^\s]+)", log_text)
    for raw_path in reversed(phase_matches):
        candidate = Path(raw_path.strip())
        if (candidate / "train_config.json").exists():
            return candidate

    phase_dirs = [
        path for path in requested_run_dir.glob("phase*") if path.is_dir() and (path / "train_config.json").exists()
    ]
    if phase_dirs:
        return max(phase_dirs, key=lambda path: path.stat().st_mtime)

    return requested_run_dir


def load_run_metadata(run_dir: Path) -> dict:
    config_path = run_dir / "train_config.json"
    if not config_path.exists():
        return {}
    try:
        return load_json(config_path)
    except Exception:
        return {}


def load_resume_timesteps(resume_model: str) -> int:
    if not resume_model:
        return 0
    try:
        model = PPO.load(resume_model, device="cpu")
        return int(model.num_timesteps)
    except Exception:
        return 0


def collect_preview_rows(previews_dir: Path) -> list[dict]:
    rows: list[dict] = []
    if not previews_dir.exists():
        return rows

    for summary_path in sorted(previews_dir.glob("step_*/summary.json")):
        try:
            data = json.loads(read_text(summary_path))
        except Exception:
            continue
        step_match = re.search(r"step_(\d+)", summary_path.as_posix())
        step_value = int(step_match.group(1)) if step_match else 0
        max_x_positions = data.get("max_x_positions") or []
        max_x = max_x_positions[0] if max_x_positions else None
        cleared = int(data.get("flags_cleared", 0))
        rows.append(
            {
                "label": f"{step_value // 1000}k",
                "step": step_value,
                "avgReturn": float(data.get("average_return", 0.0)),
                "testedSteps": int(round(float(data.get("average_length", 0.0)))),
                "cleared": cleared,
                "maxX": max_x,
                "remainingDistance": 0 if cleared > 0 else (
                    max(0, LEVEL_1_1_GOAL_X - int(max_x)) if max_x is not None else None
                ),
            }
        )
    return rows


def js(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def build_canvas_source(data: dict) -> str:
    return f"""import {{
  BarChart,
  Callout,
  Divider,
  Grid,
  H1,
  H2,
  Pill,
  Row,
  Spacer,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
}} from "cursor/canvas";

const currentRun = {js(data["currentRun"])};
const strongestFinishedRun = {js(data["strongestFinishedRun"])};
const latestTests = {js(data["latestTests"])};
const recentSpeed = {js(data["recentSpeed"])};

function SimpleProgress({{
  value,
  max,
  theme,
}}: {{
  value: number;
  max: number;
  theme: ReturnType<typeof useHostTheme>;
}}) {{
  const width = `${{Math.max(0, Math.min(100, (value / Math.max(max, 1)) * 100)).toFixed(1)}}%`;
  return (
    <div
      style={{{{
        background: theme.fill.secondary,
        border: `1px solid ${{theme.stroke.primary}}`,
        height: 24,
        width: "100%",
        overflow: "hidden",
      }}}}
    >
      <div
        style={{{{
          width,
          height: "100%",
          background: theme.accent.primary,
        }}}}
      />
    </div>
  );
}}

export default function MarioTrainingMonitor() {{
  const theme = useHostTheme();
  const runAdvance = currentRun.currentSteps - currentRun.startSteps;
  const recentFive = latestTests.slice(-5);

  return (
    <Stack
      gap={{20}}
      style={{{{
        background: theme.bg.editor,
        color: theme.text.primary,
        minHeight: "100%",
        padding: 24,
      }}}}
    >
      <Stack gap={{10}}>
        <Row gap={{10}} wrap>
          <Pill tone="info">GPU Training</Pill>
          <Pill tone={{currentRun.hasClearRecord ? "success" : "warning"}}>
            {{currentRun.hasClearRecord ? "Clear Recorded" : "No Clear Yet"}}
          </Pill>
          <Pill tone="neutral">{{currentRun.name}}</Pill>
        </Row>
        <H1>Mario Training Monitor</H1>
        <Text>Focus on three things: current progress, next preview, and remaining distance to the goal in each preview round.</Text>
      </Stack>

      <Callout tone={{currentRun.hasClearRecord ? "success" : "warning"}}>
        {{currentRun.summaryLine}}
      </Callout>

      <Grid columns={{4}} gap={{16}}>
        <Stat value={{currentRun.latestRemainingDistanceLabel}} label="Latest Distance To Goal" />
        <Stat value={{currentRun.bestRemainingDistanceLabel}} label="Best Distance To Goal" />
        <Stat value={{currentRun.latestScoreLabel}} label="Latest Shaped Score" />
        <Stat value={{currentRun.bestScoreLabel}} label="Best Shaped Score" />
      </Grid>

      <Grid columns={{2}} gap={{20}}>
        <Stack gap={{12}}>
          <H2>Run Progress</H2>
          <Text tone="secondary" size="small">
            This run resumes from the latest stronger GPU checkpoint rather than starting from zero.
          </Text>
          <SimpleProgress value={{currentRun.currentSteps}} max={{currentRun.targetSteps}} theme={{theme}} />
          <Row gap={{8}}>
            <Text tone="secondary">Start</Text>
            <Text weight="semibold">{{currentRun.startSteps.toLocaleString()}}</Text>
            <Spacer />
            <Text tone="secondary">Current</Text>
            <Text weight="semibold">{{currentRun.currentSteps.toLocaleString()}}</Text>
            <Spacer />
            <Text tone="secondary">Target</Text>
            <Text weight="semibold">{{currentRun.targetSteps.toLocaleString()}}</Text>
          </Row>
          <Row gap={{8}}>
            <Text tone="secondary">Added This Run</Text>
            <Text weight="semibold">{{runAdvance.toLocaleString()}} steps</Text>
            <Spacer />
            <Text tone="secondary">Recent Speed</Text>
            <Text weight="semibold">{{currentRun.avgFps}} FPS</Text>
          </Row>
        </Stack>

        <Stack gap={{12}}>
          <H2>Key Status</H2>
          <Table
            headers={{["Question", "Answer"]}}
            rows={{[
              ["Still training?", currentRun.status],
              ["GPU?", `${{currentRun.device}} / ${{currentRun.gpu}}`],
              ["Any clears?", currentRun.hasClearRecord ? "Yes" : "No"],
              ["Current total steps", currentRun.currentSteps.toLocaleString()],
              ["Best score so far", currentRun.bestScoreLabel],
              ["Best remaining distance", currentRun.bestRemainingDistanceLabel],
              ["Next preview point", `${{currentRun.nextPreviewAt.toLocaleString()}} steps`],
            ]}}
          />
        </Stack>
      </Grid>

      <Divider />

      <Grid columns={{2}} gap={{20}}>
        <Stack gap={{12}}>
          <H2>Distance To Goal By Preview</H2>
          <Text tone="secondary" size="small">
            Each bar shows how far Mario still was from the World 1-1 goal line in that preview.
            Goal x is set to {{currentRun.goalX}}. Smaller is better, zero means a clear.
          </Text>
          <BarChart
            categories={{latestTests.map((item) => item.label)}}
            series={{[
              {{
                name: "Remaining Distance",
                data: latestTests.map((item) => item.remainingDistance ?? currentRun.goalX),
                tone: "warning",
              }},
            ]}}
            height={{220}}
          />
        </Stack>

        <Stack gap={{12}}>
          <H2>Score By Preview</H2>
          <Text tone="secondary" size="small">
            This is the score under our own reward and penalty shaping. Higher is better.
          </Text>
          <BarChart
            categories={{latestTests.map((item) => item.label)}}
            series={{[
              {{
                name: "Shaped Score",
                data: latestTests.map((item) => item.avgReturn),
                tone: "success",
              }},
            ]}}
            height={{220}}
          />
        </Stack>
      </Grid>

      <Divider />

      <Grid columns={{2}} gap={{20}}>
        <Stack gap={{12}}>
          <H2>Recent 5 Previews</H2>
          <Table
            headers={{["Preview", "Score", "Max X", "Remaining", "Cleared"]}}
            rows={{recentFive.map((item) => [
              item.label,
              item.avgReturn.toFixed(1),
              item.maxX === null ? "n/a" : String(item.maxX),
              item.remainingDistance === null ? "n/a" : String(item.remainingDistance),
              item.cleared > 0 ? "Yes" : "No",
            ])}}
            rowTone={{recentFive.map((item) => (item.cleared > 0 ? "success" : "warning"))}}
          />
        </Stack>

        <Stack gap={{12}}>
          <H2>Previous Best Run</H2>
          <Table
            headers={{["Field", "Value"]}}
            rows={{[
              ["Run", strongestFinishedRun.name],
              ["Final steps", strongestFinishedRun.finalSteps.toLocaleString()],
              ["Elapsed", strongestFinishedRun.elapsed],
              ["Clears", String(strongestFinishedRun.flagClears)],
              ["Model", strongestFinishedRun.bestModel],
            ]}}
          />
        </Stack>
      </Grid>
    </Stack>
  );
}}
"""


def build_data(log_text: str, run_dir: Path) -> dict:
    command = parse_header_command(log_text)
    run_metadata = load_run_metadata(run_dir)
    requested_timesteps = int(run_metadata.get("timesteps") or parse_arg(command, "--timesteps") or "0")
    resume_model = str(run_metadata.get("resume_model") or parse_arg(command, "--resume-model") or "")
    start_timesteps = int(run_metadata.get("start_timesteps") or load_resume_timesteps(resume_model))
    current_steps = parse_last_int_metric(log_text, "total_timesteps") or start_timesteps
    avg_fps = parse_last_int_metric(log_text, "fps") or 0
    elapsed_seconds = parse_last_int_metric(log_text, "time_elapsed") or 0
    target_steps = start_timesteps + requested_timesteps if requested_timesteps else current_steps
    preview_config = run_metadata.get("preview") or {}
    preview_freq = int(preview_config.get("preview_freq") or parse_arg(command, "--preview-freq") or "0")
    next_preview = (
        ((current_steps // preview_freq) + 1) * preview_freq if preview_freq > 0 else current_steps
    )
    remaining_to_preview = max(0, next_preview - current_steps)
    eta_seconds = int(remaining_to_preview / max(avg_fps, 1)) if remaining_to_preview else 0
    preview_rows = collect_preview_rows(run_dir / "previews")
    has_clear_record = any(row["cleared"] > 0 for row in preview_rows)
    latest_remaining_distance = (
        preview_rows[-1]["remainingDistance"] if preview_rows and preview_rows[-1]["remainingDistance"] is not None else None
    )
    latest_score = preview_rows[-1]["avgReturn"] if preview_rows else None
    best_score = max((row["avgReturn"] for row in preview_rows), default=None)
    distance_candidates = [row["remainingDistance"] for row in preview_rows if row["remainingDistance"] is not None]
    best_remaining_distance = min(distance_candidates) if distance_candidates else None
    strongest_elapsed = "10m 42s"

    summary_line = (
        f"This run has reached {current_steps:,} steps at about {avg_fps} FPS."
        if parse_run_status(log_text) == "Training"
        else f"This run has finished at {current_steps:,} steps."
    )
    if not has_clear_record:
        summary_line += " No saved preview has shown a clear yet."
    else:
        summary_line += " At least one saved preview has cleared the level."
    if latest_remaining_distance is not None:
        summary_line += f" In the latest preview, Mario was about {latest_remaining_distance} units from the goal."
    if best_score is not None:
        summary_line += f" Best shaped score so far is {best_score:.1f}."

    recent_speed = []
    metric_matches = list(
        re.finditer(
            r"\|\s+fps\s+\|\s+([^\|]+?)\s+\|.*?\|\s+total_timesteps\s+\|\s+([^\|]+?)\s+\|",
            log_text,
            re.DOTALL,
        )
    )
    if not metric_matches:
        # fallback: parse separate metrics by block
        step_values = re.findall(r"\|\s+total_timesteps\s+\|\s+([^\|]+?)\s+\|", log_text)
        fps_values = re.findall(r"\|\s+fps\s+\|\s+([^\|]+?)\s+\|", log_text)
        for step_str, fps_str in list(zip(step_values, fps_values))[-5:]:
            recent_speed.append(
                {
                    "label": f"{int(float(step_str.strip())) // 1000}k",
                    "fps": int(float(fps_str.strip())),
                }
            )

    if not recent_speed:
        step_values = re.findall(r"\|\s+total_timesteps\s+\|\s+([^\|]+?)\s+\|", log_text)
        fps_values = re.findall(r"\|\s+fps\s+\|\s+([^\|]+?)\s+\|", log_text)
        for step_str, fps_str in list(zip(step_values, fps_values))[-5:]:
            recent_speed.append(
                {
                    "label": f"{int(float(step_str.strip())) // 1000}k",
                    "fps": int(float(fps_str.strip())),
                }
            )

    return {
        "currentRun": {
            "name": run_dir.name,
            "status": "Running" if parse_run_status(log_text) == "Training" else "Completed",
            "device": "CUDA",
            "gpu": "RTX 4070 SUPER",
            "startSteps": start_timesteps,
            "currentSteps": current_steps,
            "targetSteps": target_steps,
            "avgFps": avg_fps,
            "elapsed": f"~ {elapsed_seconds} sec",
            "nextPreviewAt": next_preview,
            "remainingToPreview": remaining_to_preview,
            "etaToPreview": f"~ {max(1, eta_seconds // 60)} min" if eta_seconds >= 60 else f"~ {eta_seconds} sec",
            "hasClearRecord": has_clear_record,
            "goalX": LEVEL_1_1_GOAL_X,
            "latestRemainingDistanceLabel": "n/a" if latest_remaining_distance is None else f"{latest_remaining_distance}",
            "bestRemainingDistanceLabel": "n/a" if best_remaining_distance is None else f"{best_remaining_distance}",
            "latestScoreLabel": "n/a" if latest_score is None else f"{latest_score:.1f}",
            "bestScoreLabel": "n/a" if best_score is None else f"{best_score:.1f}",
            "summaryLine": summary_line,
            "runDir": str(run_dir),
        },
        "strongestFinishedRun": {
            "name": "fastclear-gpu-main",
            "bestModel": "runs/fastclear-gpu-main/models/mario_final.zip",
            "finalSteps": 100352,
            "elapsed": strongest_elapsed,
            "flagClears": 0,
        },
        "latestTests": preview_rows if preview_rows else [],
        "recentSpeed": recent_speed,
    }


def write_canvas(canvas_path: Path, data: dict) -> None:
    canvas_path.write_text(build_canvas_source(data), encoding="utf-8")


def main() -> None:
    args = parse_args()
    terminal_file = Path(args.terminal_file)
    requested_run_dir = Path(args.run_dir)
    canvas_file = Path(args.canvas_file)

    last_render = ""
    while True:
        log_text = read_text(terminal_file)
        run_dir = resolve_active_run_dir(log_text, requested_run_dir)
        data = build_data(log_text, run_dir)
        rendered = build_canvas_source(data)
        if rendered != last_render:
            canvas_file.write_text(rendered, encoding="utf-8")
            last_render = rendered

        if "train_ok run_dir=" in log_text and "exit_code:" in log_text:
            break
        time.sleep(max(3, args.interval))


if __name__ == "__main__":
    main()
