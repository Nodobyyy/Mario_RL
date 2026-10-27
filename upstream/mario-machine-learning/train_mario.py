from __future__ import annotations

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import VecNormalize

from mario_runtime import ACTION_SET_MAP
from mario_runtime import EvalCheckpointCallback
from mario_runtime import EnvConfig
from mario_runtime import config_to_dict
from mario_runtime import make_vec_env
from mario_runtime import PreviewCallback
from mario_runtime import timestamped_run_dir
from mario_runtime import write_json
from rnd import RNDModule
from rnd import RNDVecEnvWrapper


def jump_action_indices(action_set: str, include_long_jump: bool = False) -> list[int]:
    """Return indices of jump-containing actions (buttons containing 'A') for the given action set.
    When `include_long_jump` is True, also include the extra macro action appended by
    LongJumpActionWrapper (which is always `len(actions)` - the last index).
    """
    actions = ACTION_SET_MAP.get(action_set, ACTION_SET_MAP["right_only"])
    idx = [i for i, buttons in enumerate(actions) if "A" in buttons]
    if include_long_jump:
        idx.append(len(actions))
    return idx


def long_jump_macro_index(action_set: str) -> int:
    """Index of the macro action appended by LongJumpActionWrapper."""
    return len(ACTION_SET_MAP.get(action_set, ACTION_SET_MAP["right_only"]))


def apply_action_logit_bias(model: PPO, indices: list[int], bias_value: float) -> None:
    """Add `bias_value` to selected actions' logit bias on the policy's action head.

    This is an "action prior" that skews rollout sampling toward the given actions without
    breaking PPO's importance sampling (log_prob is still computed from the biased policy).
    PPO is free to learn the bias away if the actions turn out to be suboptimal.
    """
    with torch.no_grad():
        for idx in indices:
            model.policy.action_net.bias[idx] += bias_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PPO agent for Super Mario Bros (single stage via --world/--stage).",
    )
    parser.add_argument("--world", type=int, default=1, help="World 1–8 (vanilla SMB).")
    parser.add_argument("--stage", type=int, default=1, help="Stage 1–4 within the world.")
    parser.add_argument(
        "--goal-line-x",
        type=int,
        default=0,
        help="Override flag-line x for x-metrics / shaping sanity (0 = use built-in table for world/stage).",
    )
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--resume-model", type=str, default="")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument(
        "--vec-backend",
        type=str,
        default="dummy",
        choices=["dummy", "subproc"],
    )
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--screen-size", type=int, default=84)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--action-set", type=str, default="right_only", choices=["right_only", "simple"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--checkpoint-freq", type=int, default=5_000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor. Mario benefits from a lower gamma (~0.9) so the value "
             "function distinguishes 'about to die in pit' from 'past the pit'.",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda for advantage estimation.",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=-1.0,
        help="Early-stop PPO epoch when approx KL exceeds this. Use -1 to disable.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--forward-reward-scale", type=float, default=0.15)
    parser.add_argument("--backward-penalty-scale", type=float, default=0.2)
    parser.add_argument("--flag-bonus", type=float, default=1000.0)
    parser.add_argument("--stall-steps", type=int, default=40)
    parser.add_argument("--stall-penalty", type=float, default=2.0)
    parser.add_argument(
        "--end-on-stall-steps",
        type=int,
        default=0,
        help="Truncate episode after this many consecutive non-forward steps. 0 disables.",
    )
    parser.add_argument(
        "--end-on-stall-penalty",
        type=float,
        default=0.0,
        help="One-time penalty applied when the episode truncates due to stall.",
    )
    parser.add_argument(
        "--milestone-step",
        type=int,
        default=0,
        help="Pixel width of each new-distance milestone. 0 disables milestone bonus.",
    )
    parser.add_argument(
        "--milestone-bonus",
        type=float,
        default=0.0,
        help="One-time bonus when Mario first reaches each new milestone within an episode.",
    )
    parser.add_argument(
        "--hurdle-x",
        type=int,
        nargs="*",
        default=[],
        help="One or more x-position thresholds. Mario gets hurdle-bonus the first time he crosses each.",
    )
    parser.add_argument(
        "--hurdle-bonus",
        type=float,
        default=0.0,
        help="One-time bonus granted per hurdle crossed within an episode.",
    )
    parser.add_argument(
        "--time-penalty-per-step",
        type=float,
        default=0.0,
        help="Subtract this from the shaped reward every agent step (encourages shorter episodes). "
        "Typical fine-tune values: 0.01–0.05 with normalize-reward.",
    )
    parser.add_argument(
        "--action-bias-jump",
        type=float,
        default=0.0,
        help="Add this value to the action_net logit bias of every jump-containing action at init. "
             "Positive values (e.g. 1.0) push initial sampling toward jumping; PPO can learn it back.",
    )
    parser.add_argument(
        "--action-bias-long-jump",
        type=float,
        default=0.0,
        help="Additional logit bias applied to the long-jump macro action only "
             "(on top of --action-bias-jump). Requires --long-jump-action.",
    )
    parser.add_argument(
        "--long-jump-action",
        action="store_true",
        help="Append a macro action that holds (right+A+B) for --long-jump-hold-steps agent steps. "
             "Each held step costs `frame_skip` raw frames, so total raw frames held = hold_steps * frame_skip.",
    )
    parser.add_argument(
        "--long-jump-hold-steps",
        type=int,
        default=4,
        help="Number of inner env steps to hold the base action during a long jump macro.",
    )
    parser.add_argument(
        "--long-jump-base-action",
        type=int,
        default=-1,
        help="Action index within the action set used by the long jump macro. -1 = auto-pick the "
             "last action that includes both 'right' and 'A' (typically right+A+B).",
    )
    parser.add_argument(
        "--normalize-reward",
        action="store_true",
        help="Wrap vec env with VecNormalize(norm_reward=True) so return scale stays in a healthy range for PPO.",
    )
    parser.add_argument(
        "--norm-reward-clip",
        type=float,
        default=10.0,
        help="Absolute clip on normalized rewards when --normalize-reward is set.",
    )
    parser.add_argument(
        "--rnd-coef",
        type=float,
        default=0.0,
        help="RND intrinsic-reward coefficient. 0 disables RND.",
    )
    parser.add_argument(
        "--rnd-lr",
        type=float,
        default=1e-4,
        help="Learning rate for the RND predictor network.",
    )
    parser.add_argument(
        "--rnd-feature-dim",
        type=int,
        default=256,
        help="Output feature dimension of the RND target/predictor networks.",
    )
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--eval-steps", type=int, default=4_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--preview-freq", type=int, default=5_000)
    parser.add_argument("--preview-steps", type=int, default=1024)
    parser.add_argument("--preview-episodes", type=int, default=1)
    parser.add_argument("--preview-fps", type=int, default=15)
    parser.add_argument("--live-preview", action="store_true")
    return parser.parse_args()


def resolve_vecnormalize_pkl(resume_model_path: Path) -> Path | None:
    """VecNormalize is saved next to checkpoints under models/; best_eval.zip lives in evaluations/."""
    direct = resume_model_path.parent / "vecnormalize.pkl"
    if direct.exists():
        return direct
    alt = resume_model_path.parent.parent / "models" / "vecnormalize.pkl"
    if alt.exists():
        return alt
    return None


def resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def resume_custom_objects(args: argparse.Namespace) -> dict[str, float | int | None]:
    custom: dict[str, float | int | None] = {
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "ent_coef": args.ent_coef,
        "n_epochs": args.n_epochs,
        "clip_range": args.clip_range,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
    }
    custom["target_kl"] = args.target_kl if args.target_kl > 0 else None
    return custom


def build_callbacks(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    config: EnvConfig,
    start_timesteps: int,
) -> list:
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq // args.n_envs),
        save_path=str(run_dir / "models" / "checkpoints"),
        name_prefix="mario_ppo",
        verbose=1,
    )
    callbacks = [checkpoint_callback]
    if args.eval_freq > 0:
        callbacks.append(
            EvalCheckpointCallback(
                eval_freq=args.eval_freq,
                output_dir=run_dir / "evaluations",
                eval_steps=args.eval_steps,
                eval_episodes=args.eval_episodes,
                deterministic=True,
                seed=args.seed,
                start_timesteps=start_timesteps,
                config=config,
                verbose=1,
            )
        )
    if args.preview_freq > 0:
        callbacks.append(
            PreviewCallback(
                preview_freq=args.preview_freq,
                output_dir=run_dir / "previews",
                preview_steps=args.preview_steps,
                preview_episodes=args.preview_episodes,
                preview_fps=args.preview_fps,
                deterministic=True,
                render_human=args.live_preview,
                seed=args.seed,
                start_timesteps=start_timesteps,
                config=config,
                verbose=1,
            )
        )
    return callbacks


def main() -> None:
    args = parse_args()
    resolved_device = resolve_device(args.device)
    resume_model_path = Path(args.resume_model) if args.resume_model else None
    run_dir = Path(args.run_dir) if args.run_dir else timestamped_run_dir()
    models_dir = run_dir / "models"
    logs_dir = run_dir / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    config = EnvConfig(
        world=args.world,
        stage=args.stage,
        goal_line_x=args.goal_line_x,
        n_envs=args.n_envs,
        vec_backend=args.vec_backend,
        frame_skip=args.frame_skip,
        frame_stack=args.frame_stack,
        screen_size=args.screen_size,
        noop_max=args.noop_max,
        end_on_flag=True,
        action_set=args.action_set,
        forward_reward_scale=args.forward_reward_scale,
        backward_penalty_scale=args.backward_penalty_scale,
        flag_bonus=args.flag_bonus,
        stall_steps=args.stall_steps,
        stall_penalty=args.stall_penalty,
        end_on_stall_steps=args.end_on_stall_steps,
        end_on_stall_penalty=args.end_on_stall_penalty,
        milestone_step=args.milestone_step,
        milestone_bonus=args.milestone_bonus,
        hurdle_x=tuple(args.hurdle_x),
        hurdle_bonus=args.hurdle_bonus,
        time_penalty_per_step=args.time_penalty_per_step,
        long_jump_action=args.long_jump_action,
        long_jump_hold_steps=args.long_jump_hold_steps,
        long_jump_base_action=args.long_jump_base_action,
    )

    env = make_vec_env(config, seed=args.seed)
    if args.rnd_coef > 0.0:
        obs_shape = env.observation_space.shape
        env = RNDVecEnvWrapper(
            env,
            RNDModule(
                obs_shape=obs_shape,
                device=resolved_device,
                feature_dim=args.rnd_feature_dim,
                lr=args.rnd_lr,
            ),
            intrinsic_coef=args.rnd_coef,
        )
    if args.normalize_reward:
        vec_pkl = resolve_vecnormalize_pkl(resume_model_path) if resume_model_path else None
        if resume_model_path and vec_pkl is not None:
            print(f"[vecnorm] loading running reward stats from {vec_pkl}")
            env = VecNormalize.load(str(vec_pkl), env)
            env.training = True
        else:
            if resume_model_path:
                print(
                    "[vecnorm] warning: no vecnormalize.pkl next to checkpoint or run/models/; "
                    "starting fresh reward normalisation (not ideal for resume).",
                )
            env = VecNormalize(
                env,
                norm_obs=False,
                norm_reward=True,
                clip_reward=args.norm_reward_clip,
                gamma=0.99,
            )
    start_timesteps = 0
    callbacks = build_callbacks(
        args=args,
        run_dir=run_dir,
        config=config,
        start_timesteps=start_timesteps,
    )

    if resume_model_path:
        model = PPO.load(
            str(resume_model_path),
            env=env,
            device=resolved_device,
            tensorboard_log=str(logs_dir / "tensorboard"),
            custom_objects=resume_custom_objects(args),
        )
        model.verbose = 1
        start_timesteps = int(model.num_timesteps)
        callbacks = build_callbacks(
            args=args,
            run_dir=run_dir,
            config=config,
            start_timesteps=start_timesteps,
        )
    else:
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            ent_coef=args.ent_coef,
            n_epochs=args.n_epochs,
            clip_range=args.clip_range,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            target_kl=args.target_kl if args.target_kl > 0 else None,
            tensorboard_log=str(logs_dir / "tensorboard"),
            device=resolved_device,
        )
        if args.action_bias_jump != 0.0:
            jump_idx = jump_action_indices(args.action_set, include_long_jump=args.long_jump_action)
            apply_action_logit_bias(model, jump_idx, args.action_bias_jump)
            print(
                f"[action-bias] Applied +{args.action_bias_jump} logit bias to jump actions "
                f"{jump_idx} (action_set={args.action_set}, long_jump={args.long_jump_action})"
            )
        if args.long_jump_action and args.action_bias_long_jump != 0.0:
            macro_idx = long_jump_macro_index(args.action_set)
            apply_action_logit_bias(model, [macro_idx], args.action_bias_long_jump)
            print(
                f"[action-bias] Applied additional +{args.action_bias_long_jump} logit bias to "
                f"long-jump macro action (index {macro_idx})"
            )
    write_json(
        {
            "run_dir": str(run_dir),
            "timesteps": args.timesteps,
            "start_timesteps": start_timesteps,
            "resume_model": str(resume_model_path) if resume_model_path else "",
            "seed": args.seed,
            "env": config_to_dict(config),
            "ppo": {
                "learning_rate": args.learning_rate,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "ent_coef": args.ent_coef,
                "n_epochs": args.n_epochs,
                "clip_range": args.clip_range,
                "vf_coef": args.vf_coef,
                "max_grad_norm": args.max_grad_norm,
                "target_kl": args.target_kl if args.target_kl > 0 else None,
                "device": resolved_device,
                "requested_device": args.device,
            },
            "rnd": {
                "rnd_coef": args.rnd_coef,
                "rnd_lr": args.rnd_lr,
                "rnd_feature_dim": args.rnd_feature_dim,
            },
            "normalize_reward": {
                "enabled": args.normalize_reward,
                "clip": args.norm_reward_clip,
            },
            "action_bias": {
                "jump": args.action_bias_jump,
                "jump_indices": jump_action_indices(args.action_set, include_long_jump=args.long_jump_action) if args.action_bias_jump != 0.0 else [],
                "long_jump": args.action_bias_long_jump,
                "long_jump_index": long_jump_macro_index(args.action_set) if args.long_jump_action else None,
            },
            "long_jump": {
                "enabled": args.long_jump_action,
                "hold_steps": args.long_jump_hold_steps,
                "base_action": args.long_jump_base_action,
            },
            "evaluation": {
                "eval_freq": args.eval_freq,
                "eval_steps": args.eval_steps,
                "eval_episodes": args.eval_episodes,
            },
            "preview": {
                "preview_freq": args.preview_freq,
                "preview_steps": args.preview_steps,
                "preview_episodes": args.preview_episodes,
                "preview_fps": args.preview_fps,
                "live_preview": args.live_preview,
            },
        },
        run_dir / "train_config.json",
    )
    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=True,
        callback=CallbackList(callbacks),
        tb_log_name="mario_ppo",
        reset_num_timesteps=not bool(resume_model_path),
    )

    final_model_path = models_dir / "mario_final.zip"
    model.save(final_model_path)
    if args.normalize_reward and isinstance(env, VecNormalize):
        env.save(str(models_dir / "vecnormalize.pkl"))
    write_json(
        {
            "run_dir": str(run_dir),
            "final_model": str(final_model_path),
            "timesteps": args.timesteps,
            "start_timesteps": start_timesteps,
            "resume_model": str(resume_model_path) if resume_model_path else "",
            "seed": args.seed,
            "env": config_to_dict(config),
            "ppo": {
                "learning_rate": args.learning_rate,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "ent_coef": args.ent_coef,
                "n_epochs": args.n_epochs,
                "clip_range": args.clip_range,
                "vf_coef": args.vf_coef,
                "max_grad_norm": args.max_grad_norm,
                "target_kl": args.target_kl if args.target_kl > 0 else None,
                "device": resolved_device,
                "requested_device": args.device,
            },
            "evaluation": {
                "eval_freq": args.eval_freq,
                "eval_steps": args.eval_steps,
                "eval_episodes": args.eval_episodes,
            },
            "preview": {
                "preview_freq": args.preview_freq,
                "preview_steps": args.preview_steps,
                "preview_episodes": args.preview_episodes,
                "preview_fps": args.preview_fps,
                "live_preview": args.live_preview,
            },
        },
        run_dir / "train_config.json",
    )
    print(f"train_ok run_dir={run_dir} final_model={final_model_path}")
    env.close()


if __name__ == "__main__":
    main()

