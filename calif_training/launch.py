"""Launch one of the five paper settings, with IF-RLVR or CALIF rewards.

Use ``--dry-run`` to inspect the command without downloading files, touching a
GPU, creating a run directory, or requiring the optional training dependencies.
The actual training command is the pinned, locally modified verl main_ppo.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import sys
from typing import Any

from calif_training.cache import DEFAULT_CACHE_DIR


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "allenai/IF_multi_constraints_upto5"
DATASET_REVISION = "2e3a77407b7fce69f95b248d64a884e3ae1c2423"
METHODS = ("constraint_only", "calif")


def settings() -> dict[str, dict[str, Any]]:
    return json.loads((REPOSITORY_ROOT / "configs" / "settings.json").read_text())


def hydra_value(value: Any) -> str:
    """Quote strings for Hydra itself, not for the shell (we pass an argv list)."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--setting", choices=tuple(settings()), required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--dry-run", action="store_true", help="print JSON; no downloads or training")
    p.add_argument("--smoke-test", action="store_true", help="one optimizer step with a tiny batch; NOT a paper result")
    p.add_argument("--gpus", type=int, help="visible GPUs on this single node (default: 8; smoke test: 1)")
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument(
        "--cpu-offload", action=argparse.BooleanOptionalAction, default=None,
        help="offload actor parameters and optimizer to CPU (paper: off; smoke test: on)",
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    p.add_argument("--actor-token-budget", type=int, default=32768)
    p.add_argument("--log-prob-token-budget", type=int, default=49152)
    p.add_argument("--rollout-max-num-seqs", type=int, default=1024)
    p.add_argument("--agent-workers", type=int, default=32)
    p.add_argument("--filter-workers", type=int, default=16)
    p.add_argument("--enforce-eager", action="store_true", help="disable vLLM CUDA graphs")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--output-dir", type=Path, help="default: outputs/<setting>_<method>")
    p.add_argument("--run-name", help="tracking label; also names the default output directory unless --output-dir is set")
    p.add_argument("--project", default="CALIF")
    p.add_argument("--wandb", action="store_true", help="enable W&B; authenticate outside this command")
    p.add_argument("--resume-from", type=Path, help="explicit global_step_* directory; never auto-resume")
    return p


def validate_args(args: argparse.Namespace) -> None:
    if args.gpus is None:
        args.gpus = 1 if args.smoke_test else 8
    for key in ("gpus", "tensor_parallel", "actor_token_budget", "log_prob_token_budget",
                "rollout_max_num_seqs", "agent_workers", "filter_workers"):
        if getattr(args, key) < 1:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if args.gpus % args.tensor_parallel:
        raise ValueError("--gpus must be divisible by --tensor-parallel")
    batch_size = 2 if args.smoke_test else 1024
    if batch_size % args.gpus:
        raise ValueError(f"The {batch_size}-prompt batch must be divisible by --gpus")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("--gpu-memory-utilization must be between 0 and 1")
    for value, name in ((args.run_name, "run-name"), (args.project, "project")):
        if value is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise ValueError(f"--{name} may contain only letters, numbers, _, -, and .")


def build_plan(
    args: argparse.Namespace,
    cache_path: Path | None = None,
    model_path: Path | None = None,
) -> dict[str, Any]:
    """Build a training plan without importing GPU dependencies."""
    from calif_training.cache import setting_spec

    validate_args(args)
    spec = settings()[args.setting]
    artifact_spec = setting_spec(args.setting)
    if spec["base_model"] != artifact_spec["base_model"]:
        raise ValueError("Training setting and artifact manifest disagree about the base model")
    model_source = str(model_path.resolve()) if model_path else spec["base_model"]
    cpu_offload = args.smoke_test if args.cpu_offload is None else args.cpu_offload
    use_anchor = args.method == "calif"
    run_name = args.run_name or f"{args.setting}_{args.method}{'_smoke' if args.smoke_test else ''}"
    output = (args.output_dir or REPOSITORY_ROOT / "outputs" / run_name).resolve()
    checkpoint_dir = output / "checkpoints"
    cache_dir = args.cache_dir.resolve()
    # Replaced by the hash-verified helper result before real execution.
    cache = str(cache_path.resolve()) if cache_path else "<downloaded-and-verified-training-cache>"
    env = {
        "MODEL_PATH": model_source,
        "IF_REF_VLLM_MODEL": model_source,
        "IF_REF_VLLM_BASE_URL": "",
        "IF_APPLY_ENABLE_THINKING_KWARG": str(spec["qwen_chat_template"]).lower(),
        "IF_REQUIRE_THINK_END_FOR_REWARD": str(spec["enable_thinking"]).lower(),
        "IF_ALLOW_MISSING_THINK_FINAL_ANSWER": str(not spec["enable_thinking"]).lower(),
        "IF_PPL_PREFIX_MODE": "standard",
        "IF_PPL_ANCHOR_FLIP_HANDLING": "strict",
        "IF_PPL_ANCHOR_FLOOR_ACTION": "zero",
        "IF_PPL_ANCHOR_FLOOR_MARGIN": "0.0",
        "IF_REF_ANCHOR_TRAIN_CACHED_ONLY": "false",
        "IF_REF_ANCHOR_CACHE_PATH": cache if use_anchor else "",
        "IF_REF_PPL_BASELINE": "0",
        "IF_REF_PPL_ANCHOR": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "NLTK_DATA": os.environ.get("NLTK_DATA", str(Path.home() / ".cache" / "calif" / "nltk_data")),
        "TORCHINDUCTOR_CACHE_DIR": os.environ.get(
            "TORCHINDUCTOR_CACHE_DIR", str(Path.home() / ".cache" / "calif" / "torchinductor")
        ),
        "TRITON_CACHE_DIR": os.environ.get(
            "TRITON_CACHE_DIR", str(Path.home() / ".cache" / "calif" / "triton")
        ),
        "TMPDIR": os.environ.get("TMPDIR", str(Path.home() / ".cache" / "calif" / "tmp")),
    }
    command = [sys.executable, "-m", "verl.trainer.main_ppo"]

    def put(key: str, value: Any) -> None:
        command.append(f"{key}={hydra_value(value)}")

    overrides = {
        "algorithm.adv_estimator": "grpo",
        "algorithm.use_kl_in_reward": False,
        "data.custom_cls.path": str(REPOSITORY_ROOT / "if_rlvr" / "if_dataset.py"),
        "data.custom_cls.name": "IFMultiConstraintsDataset",
        "data.train_files": ["if_multi_train"],
        "data.val_files": ["if_multi_val"],
        "+data.if_dataset_hf": DATASET_ID,
        "+data.if_dataset_revision": DATASET_REVISION,
        "+data.if_dataset_val_size": 512,
        "+data.if_dataset_seed": 1,
        "data.train_batch_size": 1024,
        "data.max_prompt_length": spec["max_prompt_length"],
        "data.max_response_length": spec["max_response_length"],
        "data.filter_overlong_prompts": True,
        "data.filter_overlong_prompts_workers": args.filter_workers,
        "data.truncation": "error",
        "data.prompt_key": "prompt",
        "data.reward_fn_key": "data_source",
        "actor_rollout_ref.model.path": model_source,
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.optim.lr": spec["learning_rate"],
        "actor_rollout_ref.actor.ppo_mini_batch_size": 1024,
        "actor_rollout_ref.actor.use_dynamic_bsz": True,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": args.actor_token_budget,
        "actor_rollout_ref.actor.use_kl_loss": True,
        "actor_rollout_ref.actor.kl_loss_coef": 0.001,
        "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.fsdp_config.param_offload": cpu_offload,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": cpu_offload,
        "actor_rollout_ref.actor.fsdp_config.model_dtype": "fp32",
        "actor_rollout_ref.actor.fsdp_config.dtype": "bfloat16",
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size": args.tensor_parallel,
        "actor_rollout_ref.rollout.gpu_memory_utilization": args.gpu_memory_utilization,
        "actor_rollout_ref.rollout.n": 8,
        "actor_rollout_ref.rollout.temperature": spec["temperature"],
        "actor_rollout_ref.rollout.top_p": spec["top_p"],
        "actor_rollout_ref.rollout.top_k": spec["top_k"],
        "+actor_rollout_ref.rollout.presence_penalty": 0.0,
        "actor_rollout_ref.rollout.response_length": spec["max_response_length"],
        "actor_rollout_ref.rollout.max_model_len": spec["max_prompt_length"] + spec["max_response_length"],
        "actor_rollout_ref.rollout.max_num_seqs": args.rollout_max_num_seqs,
        "actor_rollout_ref.rollout.max_num_batched_tokens": 8192,
        "actor_rollout_ref.rollout.enforce_eager": args.enforce_eager,
        "actor_rollout_ref.rollout.free_cache_engine": True,
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": True,
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": args.log_prob_token_budget,
        "actor_rollout_ref.rollout.agent.num_workers": args.agent_workers,
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz": True,
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu": args.log_prob_token_budget,
        "actor_rollout_ref.ref.fsdp_config.param_offload": True,
        "actor_rollout_ref.ref.fsdp_config.model_dtype": "fp32",
        "custom_reward_function.path": str(REPOSITORY_ROOT / "if_rlvr" / "if_reward_fn.py"),
        "custom_reward_function.name": "compute_score",
        "+if_py_given_x_reward_coeff": 0.1 if use_anchor else 0.0,
        "+if_px_given_y_reward_coeff": 0.0,
        "+if_ppl_reward_strategy": "anchor",
        "+if_ppl_anchor_reward_mode": "both",
        "+if_ppl_prefix_mode": "standard",
        "+if_ref_anchor_precompute": use_anchor,
        "+if_ref_policy_anchor_ppl": use_anchor,
        "+if_ref_anchor_precompute_batch_size": 4096,
        "+if_ref_anchor_cache_path": cache if use_anchor else None,
        "+if_ref_anchor_cache_metadata_strict": True,
        "+if_ref_anchor_skip_missing_precompute": False,
        "+if_ref_anchor_require_complete_cache": use_anchor,
        "+if_release_cache_setting": args.setting if use_anchor else None,
        "+if_ref_ppl_gate": False,
        "+clipped_rollout_mode": "use",
        "trainer.balance_batch": True,
        "trainer.logger": ["console", "wandb"] if args.wandb else ["console"],
        "trainer.project_name": args.project,
        "trainer.experiment_name": run_name,
        "trainer.n_gpus_per_node": args.gpus,
        "trainer.nnodes": 1,
        "trainer.save_freq": 91,
        "trainer.test_freq": -1,
        "trainer.total_epochs": spec["epochs"],
        "trainer.val_before_train": False,
        "trainer.default_local_dir": str(checkpoint_dir),
        "trainer.resume_mode": "resume_path" if args.resume_from else "disable",
        "trainer.resume_from_path": str(args.resume_from.resolve()) if args.resume_from else None,
    }
    if spec["qwen_chat_template"]:
        overrides["+data.apply_chat_template_kwargs.enable_thinking"] = spec["enable_thinking"]
    if args.smoke_test:
        overrides.update({
            "data.train_batch_size": 2,
            "data.train_max_samples": 8,
            "data.val_max_samples": 2,
            "data.shuffle": False,
            "data.dataloader_num_workers": 0,
            "data.filter_overlong_prompts_workers": 1,
            "data.max_response_length": 128,
            "actor_rollout_ref.actor.ppo_mini_batch_size": 2,
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 4096,
            "actor_rollout_ref.rollout.n": 2,
            "actor_rollout_ref.rollout.response_length": 128,
            "actor_rollout_ref.rollout.max_model_len": spec["max_prompt_length"] + 128,
            "actor_rollout_ref.rollout.gpu_memory_utilization": 0.3,
            "actor_rollout_ref.rollout.max_num_seqs": 8,
            "actor_rollout_ref.rollout.enforce_eager": True,
            "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": 8192,
            "actor_rollout_ref.rollout.agent.num_workers": 2,
            "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu": 8192,
            "trainer.logger": ["console"],
            "trainer.total_epochs": 1,
            "trainer.total_training_steps": 1,
            "trainer.save_freq": -1,
        })
    for key, value in overrides.items():
        put(key, value)
    # Ray workers must receive the same IF parsing/cache configuration as the driver.
    for key, value in env.items():
        put(f"++ray_kwargs.ray_init.runtime_env.env_vars.{key}", value)
    return {
        "setting": args.setting,
        "method": args.method,
        "profile": "smoke_test_not_paper" if args.smoke_test else "paper",
        "model": spec["base_model"],
        "model_revision": artifact_spec["base_model_revision"],
        "model_source": model_source,
        "model_snapshot_prepared": model_path is not None,
        "paper_setting": spec,
        "dataset_revision": DATASET_REVISION,
        "cache_dir": str(cache_dir),
        "output_dir": str(output),
        "environment": env,
        "command": command,
        "shell_command": shlex.join(command),
    }


def _prepare_nltk(data_dir: str) -> None:
    import nltk

    # NLTK rejects downloads into group/world-writable directories. Keep its
    # resources outside shared project/checkpoint directories by default.
    Path(data_dir).mkdir(mode=0o700, parents=True, exist_ok=True)
    nltk.data.path.insert(0, data_dir)
    for package, resource in (("punkt", "tokenizers/punkt/english.pickle"),
                              ("punkt_tab", "tokenizers/punkt_tab/english/")):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, download_dir=data_dir, quiet=True, raise_on_error=True)
            nltk.data.find(resource)


def execute(args: argparse.Namespace) -> None:
    import fcntl
    import torch

    if torch.cuda.device_count() < args.gpus:
        raise RuntimeError(f"Requested {args.gpus} visible GPUs, found {torch.cuda.device_count()}")
    if args.resume_from and not args.resume_from.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {args.resume_from}")
    plan = build_plan(args)
    for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TMPDIR"):
        Path(plan["environment"][key]).mkdir(mode=0o700, parents=True, exist_ok=True)
    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    lock_file = (output / ".train.lock").open("a")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not args.resume_from and any((output / "checkpoints").glob("global_step_*")):
        raise RuntimeError("Output already has checkpoints; use a new output directory or --resume-from")
    from calif_training.model import prepare_model

    model_path = prepare_model(args.setting)
    cache_path = None
    if args.method == "calif":
        from calif_training.cache import prepare_cache

        cache_path = prepare_cache(args.setting, args.cache_dir.resolve())
    plan = build_plan(args, cache_path=cache_path, model_path=model_path)
    _prepare_nltk(plan["environment"]["NLTK_DATA"])
    # Do not inherit unrelated IF_* flags (subset filters, reward modes,
    # old cache paths, or external verifier endpoints) from the launching shell.
    env = {key: value for key, value in os.environ.items() if not key.startswith("IF_")}
    env.update(plan["environment"])
    env["PYTHONPATH"] = str(REPOSITORY_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.pop("RAY_ADDRESS", None)
    manifest = output / f"launch-{os.getpid()}.json"
    # Store only the explicit non-secret environment above, not API keys/auth.
    manifest.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Launch manifest: {manifest}", flush=True)
    print(plan["shell_command"], flush=True)
    os.chdir(REPOSITORY_ROOT)
    # Preserve the per-run lock across exec; no automatic retry or restart.
    os.set_inheritable(lock_file.fileno(), True)
    os.execvpe(plan["command"][0], plan["command"], env)


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    try:
        validate_args(args)
        if args.dry_run:
            print(json.dumps(build_plan(args), indent=2))
        else:
            execute(args)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
