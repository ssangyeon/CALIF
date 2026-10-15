"""CPU-only tests of the published-cache fail-closed boundary."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from calif_training import cache, model


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'anchor.json'
        self.item = {'y0': [1], 'y1': [2], 'ref0_nll': 0.3, 'ref1_nll': 0.4,
                     'ref0_token_count': 1, 'ref1_token_count': 1,
                     'ref0_ppl': 1.35, 'ref1_ppl': 1.49}
        self.meta = {'model_path': 'Qwen/Qwen3-4B', 'tokenizer_class': 'Qwen2TokenizerFast',
                     'if_dataset_seed': 1, 'if_dataset_val_size': 512,
                     'train_files': ['if_multi_train'], 'max_prompt_length': 2048,
                     'apply_chat_template_kwargs': {'enable_thinking': False}}
        self.payload = {'metadata': self.meta, 'items': {'0': self.item}}
        self.spec = {'entry_count': 1,
                     'base_model': 'Qwen/Qwen3-4B',
                     'dataset_id': 'allenai/IF_multi_constraints_upto5', 'dataset_revision': 'rev'}
        self.runtime = dict(self.meta, if_dataset_hf='', if_dataset_revision='rev',
                            ppl_prefix_mode='standard', ppl_nll_scope='final_answer_tokens_only',
                            train_sample_count=1)
        self.snapshot = Path(self.tmp.name) / 'models--Qwen--Qwen3-4B' / 'snapshots' / ('a' * 40)
        self.snapshot.mkdir(parents=True)
        contents = b'{}'
        (self.snapshot / 'config.json').write_bytes(contents)
        self.model_spec = {'repo_id': 'Qwen/Qwen3-4B', 'revision': 'a' * 40, 'files': [
            {'filename': 'config.json', 'size_bytes': 2,
             'git_blob_sha1': hashlib.sha1(b'blob 2\0' + contents).hexdigest()}]}
        self.runtime['model_path'] = str(self.snapshot)
        self.write_fixture()

    def tearDown(self):
        self.tmp.cleanup()

    def write_fixture(self):
        self.path.write_text(json.dumps(self.payload))
        self.spec['size_bytes'] = self.path.stat().st_size
        self.spec['sha256'] = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def validate(self, indices=(0,), runtime=None):
        with patch.object(cache, 'setting_spec', return_value=self.spec), \
                patch.object(model, 'model_spec', return_value=self.model_spec):
            return cache.validate_for_training(self.path, 'fixture', runtime or self.runtime, indices)

    def test_valid_cache(self):
        self.assertEqual(self.validate()['items'], self.payload['items'])

    def test_checksum_failure(self):
        self.path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.validate()

    def test_missing_file(self):
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.validate()

    def test_equal_count_different_indices_rejected(self):
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.validate([1])

    def test_empty_and_duplicate_indices_rejected(self):
        for indices in ([], [0, 0]):
            with self.assertRaisesRegex(ValueError, 'nonempty and unique'):
                self.validate(indices)

    def test_runtime_identity_mismatch(self):
        for field, value in [('model_path', '/some/local/model'), ('tokenizer_class', 'OtherTokenizer'),
                             ('if_dataset_seed', 2), ('if_dataset_val_size', 16),
                             ('max_prompt_length', 8192), ('if_dataset_revision', 'new-revision'),
                             ('ppl_prefix_mode', 'empty_think'),
                             ('apply_chat_template_kwargs', {'enable_thinking': True})]:
            with self.subTest(field=field):
                runtime = dict(self.runtime, **{field: value})
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    self.validate(runtime=runtime)

    def test_generation_parameters_may_differ(self):
        runtime = dict(self.runtime, max_response_length=8192, response_length=8192,
                       rollout_top_p=0.95, rollout_temperature=0.6)
        self.assertEqual(self.validate(runtime=runtime)['items'], self.payload['items'])

    def test_floating_model_id_rejected(self):
        with self.assertRaisesRegex(ValueError, 'mismatch for model_path'):
            self.validate(runtime=dict(self.runtime, model_path='Qwen/Qwen3-4B'))

    def test_source_metadata_mismatch(self):
        self.payload['metadata'] = dict(self.meta, if_dataset_seed=2)
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            self.validate()

    def test_undeclared_incomplete_anchor_rejected(self):
        self.payload['items']['0']['y1'] = []
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, 'incomplete cache'):
            self.validate()

    def test_materialized_inverted_scores_are_loaded_unchanged(self):
        self.item.update(y1=[1], ref1_nll=0.15, ref1_ppl=1.161834242728283)
        self.write_fixture()
        before = self.path.read_bytes()
        self.assertEqual(self.validate()['items'], self.payload['items'])
        self.assertEqual(self.path.read_bytes(), before)

    def test_nonfinite_scores_rejected(self):
        for field, value in [('ref0_nll', float('nan')), ('ref1_nll', float('inf')),
                             ('ref0_token_count', 0), ('ref1_ppl', float('inf')),
                             ('y0', [-1]), ('y1', [False])]:
            with self.subTest(field=field):
                altered = copy.deepcopy(self.item)
                altered[field] = value
                self.assertFalse(cache._complete(altered))

    def test_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'count disagree'):
            self.validate(runtime=dict(self.runtime, train_sample_count=2))

    def test_other_dataset_rejected(self):
        with self.assertRaisesRegex(ValueError, 'different training dataset'):
            self.validate(runtime=dict(self.runtime, if_dataset_hf='someone/different'))

    def test_manifest_settings(self):
        manifest = cache._manifest()
        self.assertEqual(len(manifest['settings']), 5)
        self.assertEqual(manifest['repo_id'], 'sangyon/CALIF')
        self.assertEqual(len(manifest['revision']), 40)
        for spec in manifest['settings'].values():
            self.assertNotIn('empty_y1_train_indices', spec)
            self.assertNotIn('complete_entry_count', spec)
            self.assertNotIn('source_metadata', spec)
        with self.assertRaises(ValueError):
            cache.setting_spec('does-not-exist')


if __name__ == '__main__':
    unittest.main()
