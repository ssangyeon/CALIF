"""CPU checks for the reward loop used by the training launchers."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class RewardLoopTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse((ROOT / 'verl/experimental/reward_loop/reward_loop.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RewardLoopManager')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_rm_score')
        namespace = {
            'ray': SimpleNamespace(get=lambda values: values), 'np': np,
            'TensorDict': lambda values, batch_size: values,
            'DataProto': lambda **kwargs: SimpleNamespace(**kwargs),
        }
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), '<reward-loop-test>', 'exec'), namespace)
        self.compute = namespace['compute_rm_score']
        self.worker = SimpleNamespace(compute_score_batch=SimpleNamespace(remote=Mock(return_value=[
            {'reward_score': 0.75, 'reward_extra_info': {'acc': 0.75}}
        ])))
        self.manager = SimpleNamespace(
            reward_model_manager=None, reward_loop_workers=[self.worker],
            reward_manager_cls=SimpleNamespace(assemble_rm_scores=lambda data, scores: scores),
        )
        class Batch:
            def __len__(self):
                return 1

            def chunk(self, count):
                return ['chunk']

        self.data = Batch()

    def test_rule_based_reward_and_metrics_preserved(self):
        result = self.compute(self.manager, self.data)
        self.assertEqual(result.batch['rm_scores'], [0.75])
        self.assertEqual(result.non_tensor_batch['acc'].tolist(), [0.75])
        self.assertEqual(result.meta_info['reward_extra_keys'], ['acc'])

    def test_reward_model_sleep_is_preserved_on_failure(self):
        model = Mock()
        self.manager.reward_model_manager = model
        self.worker.compute_score_batch.remote.side_effect = RuntimeError('worker failed')
        with self.assertRaisesRegex(RuntimeError, 'worker failed'):
            self.compute(self.manager, self.data)
        model.wake_up.assert_called_once_with()
        model.sleep.assert_called_once_with()

    def test_external_judge_paths_absent(self):
        for relative in ('verl/trainer/ppo/ray_trainer.py', 'verl/experimental/reward_loop/reward_loop.py'):
            source = (ROOT / relative).read_text().lower()
            for term in ('llm_verifier', 'intentcheck', 'g-eval', 'external_verifier'):
                self.assertNotIn(term, source)


if __name__ == '__main__':
    unittest.main()
