"""Offline tests: constructing training commands must not import GPU libraries."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from calif_training.launch import build_plan, parser, settings


def overrides(plan):
    return {key: json.loads(value) for key, value in
            (arg.split("=", 1) for arg in plan["command"][3:])}


class LauncherTests(unittest.TestCase):
    def test_shared_cache_default(self):
        from calif_training.cache import DEFAULT_CACHE_DIR, main
        args = parser().parse_args(["--setting", "qwen3_4b_think", "--method", "calif"])
        self.assertEqual(args.cache_dir, DEFAULT_CACHE_DIR)
        self.assertTrue(DEFAULT_CACHE_DIR.is_absolute())
        with patch("calif_training.cache.prepare_cache") as prepare, patch("builtins.print"):
            main(["qwen3_4b_think"])
        prepare.assert_called_once_with("qwen3_4b_think", DEFAULT_CACHE_DIR)

    def test_run_name_and_explicit_output(self):
        argv = ["--setting", "qwen3_4b_think", "--method", "calif", "--run-name", "custom_run"]
        plan = build_plan(parser().parse_args(argv))
        self.assertEqual(Path(plan["output_dir"]).name, "custom_run")
        explicit = Path("/tmp/explicit_training_output").resolve()
        plan = build_plan(parser().parse_args(argv + ["--output-dir", str(explicit)]))
        self.assertEqual(plan["output_dir"], str(explicit))
        self.assertEqual(overrides(plan)["trainer.experiment_name"], "custom_run")

    def test_all_ten_paper_plans(self):
        self.assertEqual(len(settings()), 5)
        for setting, spec in settings().items():
            for method in ("constraint_only", "calif"):
                with self.subTest(setting=setting, method=method):
                    args = parser().parse_args(["--setting", setting, "--method", method, "--dry-run"])
                    plan = build_plan(args)
                    cfg = overrides(plan)
                    self.assertEqual(plan["profile"], "paper")
                    self.assertEqual(cfg["trainer.n_gpus_per_node"], 8)
                    self.assertEqual(cfg["data.train_batch_size"], 1024)
                    self.assertEqual(cfg["actor_rollout_ref.rollout.n"], 8)
                    self.assertEqual(cfg["actor_rollout_ref.actor.kl_loss_coef"], 0.001)
                    self.assertFalse(cfg["actor_rollout_ref.actor.fsdp_config.param_offload"])
                    self.assertFalse(cfg["actor_rollout_ref.actor.fsdp_config.optimizer_offload"])
                    self.assertEqual(cfg["trainer.total_epochs"], spec["epochs"])
                    self.assertEqual(cfg["actor_rollout_ref.model.path"], spec["base_model"])
                    self.assertEqual(plan["model_source"], spec["base_model"])
                    self.assertEqual(len(plan["model_revision"]), 40)
                    self.assertFalse(plan["model_snapshot_prepared"])
                    self.assertNotIn("data.dataloader_num_workers", cfg)
                    self.assertEqual(cfg["trainer.resume_mode"], "disable")
                    self.assertEqual(cfg["trainer.test_freq"], -1)
                    self.assertFalse(cfg["trainer.val_before_train"])
                    self.assertEqual(cfg["+if_py_given_x_reward_coeff"], 0.1 if method == "calif" else 0.0)
                    self.assertEqual(cfg["+if_ref_policy_anchor_ppl"], method == "calif")
                    self.assertEqual(cfg["+if_ref_anchor_require_complete_cache"], method == "calif")
                    self.assertFalse(cfg["+if_ref_anchor_skip_missing_precompute"])
                    self.assertEqual(plan["environment"]["IF_REF_ANCHOR_TRAIN_CACHED_ONLY"], "false")
                    self.assertEqual(plan["environment"]["IF_PPL_ANCHOR_FLIP_HANDLING"], "strict")
                    self.assertEqual(plan["environment"]["IF_PPL_ANCHOR_FLOOR_ACTION"], "zero")
                    self.assertNotIn("/NHNHOME", plan["shell_command"])

    def test_thinking_flags(self):
        for setting, spec in settings().items():
            args = parser().parse_args(["--setting", setting, "--method", "calif"])
            cfg = overrides(build_plan(args))
            key = "+data.apply_chat_template_kwargs.enable_thinking"
            if spec["qwen_chat_template"]:
                self.assertEqual(cfg[key], spec["enable_thinking"])
            else:
                self.assertNotIn(key, cfg)

    def test_explicit_resume_only(self):
        args = parser().parse_args(["--setting", "qwen3_4b_think", "--method", "calif",
                                   "--resume-from", "/tmp/example/global_step_91"])
        cfg = overrides(build_plan(args))
        self.assertEqual(cfg["trainer.resume_mode"], "resume_path")
        self.assertEqual(cfg["trainer.resume_from_path"], str(Path("/tmp/example/global_step_91").resolve()))

    def test_resolved_cache_and_space_safe_paths(self):
        args = parser().parse_args(["--setting", "qwen3_4b_think", "--method", "calif",
                                   "--output-dir", "/tmp/path with spaces"])
        cfg = overrides(build_plan(args, Path("/tmp/cache with spaces.json")))
        self.assertEqual(cfg["+if_ref_anchor_cache_path"], str(Path("/tmp/cache with spaces.json").resolve()))
        self.assertEqual(cfg["trainer.default_local_dir"], str(Path("/tmp/path with spaces/checkpoints").resolve()))

    def test_no_anchor_based_data_exclusions_for_either_method(self):
        for setting, spec in settings().items():
            self.assertNotIn("excluded_train_indices", spec)
            for method in ("constraint_only", "calif"):
                args = parser().parse_args(["--setting", setting, "--method", method])
                self.assertNotIn("+data.if_dataset_excluded_indices", overrides(build_plan(args)))

    def test_prepared_snapshot_is_used_for_model_and_workers(self):
        for method in ("constraint_only", "calif"):
            args = parser().parse_args(["--setting", "qwen3_4b_think", "--method", method])
            path = Path("/tmp/pinned model/snapshots/1cfa9a7208912126459214e8b04321603b3df60c")
            plan = build_plan(args, model_path=path)
            cfg = overrides(plan)
            self.assertTrue(plan["model_snapshot_prepared"])
            self.assertEqual(plan["model"], "Qwen/Qwen3-4B")
            self.assertEqual(plan["model_source"], str(path.resolve()))
            self.assertEqual(plan["model_revision"], "1cfa9a7208912126459214e8b04321603b3df60c")
            self.assertEqual(cfg["actor_rollout_ref.model.path"], str(path.resolve()))
            for key in ("MODEL_PATH", "IF_REF_VLLM_MODEL"):
                self.assertEqual(plan["environment"][key], str(path.resolve()))
                self.assertEqual(cfg[f"++ray_kwargs.ray_init.runtime_env.env_vars.{key}"], str(path.resolve()))

    def test_smoke_profiles_are_separate_and_small(self):
        for setting in settings():
            for method in ("constraint_only", "calif"):
                args = parser().parse_args(["--setting", setting, "--method", method, "--smoke-test"])
                plan = build_plan(args)
                cfg = overrides(plan)
                self.assertEqual(plan["profile"], "smoke_test_not_paper")
                self.assertEqual(cfg["trainer.total_training_steps"], 1)
                self.assertEqual(cfg["trainer.n_gpus_per_node"], 1)
                self.assertEqual(cfg["data.train_batch_size"], 2)
                self.assertEqual(cfg["data.dataloader_num_workers"], 0)
                self.assertEqual(cfg["actor_rollout_ref.rollout.n"], 2)
                self.assertTrue(cfg["actor_rollout_ref.actor.fsdp_config.param_offload"])
                self.assertTrue(cfg["actor_rollout_ref.actor.fsdp_config.optimizer_offload"])
                self.assertTrue(plan["output_dir"].endswith("_smoke"))
                self.assertEqual(cfg["+if_ref_anchor_require_complete_cache"], method == "calif")

    def test_invalid_hardware_fails(self):
        args = parser().parse_args(["--setting", "tulu3_8b_dpo", "--method", "constraint_only", "--gpus", "3"])
        with self.assertRaises(ValueError):
            build_plan(args)

    def test_user_runtime_cache_paths_are_forwarded(self):
        args = parser().parse_args(["--setting", "qwen3_4b_think", "--method", "calif"])
        runtime_paths = {
            "TORCHINDUCTOR_CACHE_DIR": "/tmp/test_inductor",
            "TRITON_CACHE_DIR": "/tmp/test_triton",
            "TMPDIR": "/tmp/test_working",
        }
        with patch.dict("os.environ", runtime_paths):
            plan = build_plan(args)
        cfg = overrides(plan)
        for key, value in runtime_paths.items():
            self.assertEqual(plan["environment"][key], value)
            self.assertEqual(cfg[f"++ray_kwargs.ray_init.runtime_env.env_vars.{key}"], value)

    def test_explicit_cpu_offload_changes_placement_only(self):
        for setting in settings():
            for method in ("constraint_only", "calif"):
                for smoke in (False, True):
                    argv = ["--setting", setting, "--method", method]
                    if smoke:
                        argv.append("--smoke-test")
                    baseline = overrides(build_plan(parser().parse_args(argv)))
                    for flag, expected in (("--cpu-offload", True), ("--no-cpu-offload", False)):
                        cfg = overrides(build_plan(parser().parse_args(argv + [flag])))
                        for key in ("actor_rollout_ref.actor.fsdp_config.param_offload",
                                    "actor_rollout_ref.actor.fsdp_config.optimizer_offload"):
                            self.assertEqual(cfg[key], expected)
                            cfg.pop(key)
                        unchanged = {key: value for key, value in baseline.items() if key not in (
                            "actor_rollout_ref.actor.fsdp_config.param_offload",
                            "actor_rollout_ref.actor.fsdp_config.optimizer_offload",
                        )}
                        self.assertEqual(cfg, unchanged)


if __name__ == "__main__":
    unittest.main()
