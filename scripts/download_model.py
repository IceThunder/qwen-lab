"""Install the default Qwen3.5-4B MLX snapshot without bundling any weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = 'mlx-community/Qwen3.5-4B-4bit'
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')


def main():
    from huggingface_hub import HfApi, snapshot_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', help='Optional Hugging Face commit/tag; resolved to a fixed commit')
    args = parser.parse_args()
    destination = ROOT / 'models/qwen3.5-4b'
    if destination.exists():
        raise SystemExit('models/qwen3.5-4b already exists. Keep it, or move it aside before reinstalling.')
    stage = ROOT / 'work/default-model'
    stage.mkdir(parents=True, exist_ok=True)
    manifest = stage / 'source.json'
    if manifest.exists():
        source = json.loads(manifest.read_text())
        if args.revision and args.revision != source['revision']:
            raise SystemExit('A pinned download is in progress. Omit --revision to resume it.')
    else:
        info = HfApi().model_info(REPO, revision=args.revision)
        source = {'repo_id': REPO, 'revision': info.sha, 'format': 'MLX 4-bit'}
        manifest.write_text(json.dumps(source, indent=2) + '\n')
    info = HfApi().model_info(REPO, revision=source['revision'], files_metadata=True)
    weights = stage / 'weights'
    print(f"Downloading {REPO}@{source['revision']} (resume supported)", flush=True)
    snapshot_download(REPO, revision=source['revision'], local_dir=weights, max_workers=4,
                      allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja', 'LICENSE*'])
    for required in ['config.json', 'tokenizer.json', 'tokenizer_config.json']:
        if not (weights / required).is_file():
            raise RuntimeError(f'Missing required file: {required}')
    if not list(weights.glob('*.safetensors')):
        raise RuntimeError('No model weights found')
    verified = []
    for item in info.siblings:
        path = weights / item.rfilename
        if not path.is_file():
            if item.rfilename.endswith('.safetensors'):
                raise RuntimeError(f'Missing weight: {item.rfilename}')
            continue
        if item.size is not None and path.stat().st_size != item.size:
            raise RuntimeError(f'Size mismatch: {item.rfilename}')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        expected = getattr(item.lfs, 'sha256', None) if item.lfs else None
        if expected and digest != expected:
            raise RuntimeError(f'SHA256 mismatch: {item.rfilename}')
        verified.append({'file': item.rfilename, 'size': path.stat().st_size, 'sha256': digest})
    (stage / 'checksum.json').write_text(json.dumps({'files': verified}, indent=2) + '\n')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(destination)
    print(f'Installed: {destination}\nStart the UI: bash scripts/web.sh', flush=True)


if __name__ == '__main__':
    main()
