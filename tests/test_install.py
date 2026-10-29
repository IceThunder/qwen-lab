"""Fresh-install checks without downloading model weights or requiring MLX."""
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'web'))
import backend
import model_manager as mm
spec = importlib.util.spec_from_file_location('default_download', ROOT / 'scripts/download_model.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


class InstallTests(unittest.TestCase):
    def test_empty_model_directory(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(backend, 'ROOT', Path(tmp)):
            self.assertEqual(backend.discover_models(), [])

    def test_default_mlx_id_and_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = mm.ModelManager(tmp)
            try:
                files = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors']
                meta = dict(sha='a'*40, siblings=[dict(rfilename=n, size=3, blobId='a'*40) for n in files])
                config = dict(model_type='qwen3_5', quantization=dict(bits=4), text_config=dict(num_hidden_layers=32, intermediate_size=9216))
                with patch.object(mm, 'json_get', side_effect=[meta, config]):
                    result = manager.inspect(download.REPO)
                token = result['options'][0]['id']
                self.assertEqual(manager.plans[token]['model_id'], 'qwen3.5-4b')
                (Path(tmp) / 'models/qwen3.5-4b').mkdir()
                with patch.object(mm, 'json_get', side_effect=[meta, config]):
                    manager.inspect(download.REPO)
                self.assertNotEqual(manager.plans[token]['model_id'], 'qwen3.5-4b')
            finally:
                manager.close()

    def test_download_verifies_then_publishes_and_refuses_overwrite(self):
        self.check_download(corrupt=False)

    def test_corrupt_download_stays_out_of_model_catalog(self):
        self.check_download(corrupt=True)

    def check_download(self, corrupt):
        files = {'config.json': b'{}', 'tokenizer.json': b'{}', 'tokenizer_config.json': b'{}', 'model.safetensors': b'test-weights'}
        info = SimpleNamespace(sha='a'*40, siblings=[SimpleNamespace(rfilename=name, size=len(raw), lfs=SimpleNamespace(sha256=hashlib.sha256(raw).hexdigest())) for name, raw in files.items()])
        def snapshot(repo, revision, local_dir, **kwargs):
            self.assertEqual(revision, 'a'*40)
            local_dir.mkdir(parents=True)
            for name, raw in files.items():
                (local_dir/name).write_bytes(b'x'*len(raw) if corrupt and name.endswith('.safetensors') else raw)
        hub = SimpleNamespace(HfApi=lambda: SimpleNamespace(model_info=lambda *a, **kw: info), snapshot_download=snapshot)
        with tempfile.TemporaryDirectory() as tmp, patch.object(download, 'ROOT', Path(tmp)), patch.dict(sys.modules, {'huggingface_hub': hub}), patch.object(sys, 'argv', ['download_model.py']):
            dest = Path(tmp)/'models/qwen3.5-4b'
            if corrupt:
                with self.assertRaisesRegex(RuntimeError, 'SHA256 mismatch'):
                    download.main()
                self.assertFalse(dest.exists())
                self.assertTrue((Path(tmp)/'work/default-model/source.json').exists())
            else:
                download.main()
                self.assertEqual((dest/'weights/model.safetensors').read_bytes(), files['model.safetensors'])
                self.assertEqual(json.loads((dest/'source.json').read_text())['revision'], 'a'*40)
                with self.assertRaises(SystemExit):
                    download.main()


if __name__ == '__main__':
    unittest.main()
