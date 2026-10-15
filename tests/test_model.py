"""Offline tests of the pinned Hugging Face model boundary."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from calif_training import model


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.snapshot = Path(self.tmp.name) / 'models--test--model' / 'snapshots' / ('a' * 40)
        self.snapshot.mkdir(parents=True)
        self.spec = {'repo_id': 'test/model', 'revision': 'a' * 40, 'files': []}
        for filename, contents, algo in [('config.json', b'{"model_type":"test"}', 'git_blob_sha1'),
                                         ('model.safetensors', b'FAKE WEIGHTS', 'sha256'),
                                         ('tokenizer.json', b'{}', 'git_blob_sha1')]:
            path = self.snapshot / filename
            path.write_bytes(contents)
            if algo == 'sha256':
                digest = hashlib.sha256(contents).hexdigest()
            else:
                digest = hashlib.sha1(f'blob {len(contents)}\0'.encode() + contents).hexdigest()
            self.spec['files'].append({'filename': filename, 'size_bytes': len(contents), algo: digest})

    def tearDown(self):
        self.tmp.cleanup()

    def validate(self, path=None):
        with patch.object(model, 'model_spec', return_value=self.spec):
            return model.validate_model_path(path or self.snapshot, 'test')

    def test_exact_snapshot(self):
        self.assertEqual(self.validate(), 'test/model')

    def test_floating_repo_id_not_accepted(self):
        with self.assertRaises((OSError, ValueError)):
            self.validate('test/model')

    def test_wrong_revision_rejected(self):
        moved = self.snapshot.with_name('b' * 40)
        self.snapshot.rename(moved)
        with self.assertRaisesRegex(ValueError, 'pinned repository'):
            self.validate(moved)

    def test_wrong_repository_rejected(self):
        other = self.snapshot.parent.parent.with_name('models--other--model')
        self.snapshot.parent.parent.rename(other)
        with self.assertRaisesRegex(ValueError, 'pinned repository'):
            self.validate(other / 'snapshots' / ('a' * 40))

    def test_same_size_mutation_rejected(self):
        (self.snapshot / 'model.safetensors').write_bytes(b'FAKE EVIL!!!')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.validate()

    def test_missing_file_rejected(self):
        (self.snapshot / 'config.json').unlink()
        with self.assertRaisesRegex(ValueError, 'missing or wrong size'):
            self.validate()

    def test_unpinned_tokenizer_extension_rejected(self):
        (self.snapshot / 'added_tokens.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Unpinned inference'):
            self.validate()

    def test_hf_blob_symlinks_supported_and_verified(self):
        source = self.snapshot / 'model.safetensors'
        blob = self.snapshot.parent.parent / 'blobs' / 'known-digest'
        blob.parent.mkdir()
        source.replace(blob)
        source.symlink_to(blob)
        self.assertEqual(self.validate(), 'test/model')
        blob.write_bytes(b'FAKE EVIL!!!')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.validate()

    def test_all_profiles_resolve_offline(self):
        for setting in ['qwen3_1_7b_nonthink', 'qwen3_1_7b_think', 'qwen3_4b_nonthink',
                        'qwen3_4b_think', 'tulu3_8b_dpo']:
            spec = model.model_spec(setting)
            self.assertEqual(len(spec['revision']), 40)
            self.assertTrue(any(x['filename'].endswith('.safetensors') for x in spec['files']))
            self.assertTrue(all(x.get('sha256') or x.get('git_blob_sha1') for x in spec['files']))


if __name__ == '__main__':
    unittest.main()
