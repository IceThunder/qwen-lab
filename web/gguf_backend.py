"""Project-local llama.cpp process; text chat only, no MLX activation hooks."""
import json
import os
import shutil
import secrets
import socket
import subprocess
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def specification(folder):
    meta = json.loads((folder / 'runtime.json').read_text())
    path = (folder / 'weights' / meta['file']).resolve()
    if not path.is_relative_to((folder / 'weights').resolve()) or path.suffix != '.gguf':
        raise ValueError('Invalid GGUF path')
    if not path.is_file() or path.stat().st_size != meta['size']:
        raise ValueError('GGUF download incomplete')
    return meta, path


class GGUFEngine:
    def __init__(self):
        self.process = None
        self.key = None
        self.log = None

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = self.key = None
        if self.log:
            self.log.close()
            self.log = None

    def call(self, endpoint, body=None, timeout=120):
        request = urllib.request.Request(self.url + endpoint,
            data=None if body is None else json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.secret})
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if endpoint == '/health':
                raise
            detail = exc.read(4096).decode('utf-8', errors='replace')
            try:
                error = json.loads(detail).get('error', detail)
                detail = error.get('message', str(error)) if isinstance(error, dict) else str(error)
            except (ValueError, AttributeError):
                pass
            raise RuntimeError(f'llama.cpp 请求失败（HTTP {exc.code}）：{detail[:1000]}') from exc

    def start(self, model_id, check):
        if self.key == model_id and self.process and self.process.poll() is None:
            return
        self.stop()
        _, path = specification(ROOT / 'models' / model_id)
        configured = os.environ.get('LLAMA_SERVER')
        binaries = [Path(configured)] if configured else sorted((ROOT / '.runtime/llama.cpp').rglob('llama-server'))
        if not binaries and shutil.which('llama-server'):
            binaries = [Path(shutil.which('llama-server'))]
        if not binaries or not binaries[0].is_file():
            raise RuntimeError('llama-server not found. Install llama.cpp or set LLAMA_SERVER to its executable path.')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        self.secret = secrets.token_hex(24)
        self.url = f'http://127.0.0.1:{port}'
        (ROOT / 'logs').mkdir(parents=True, exist_ok=True)
        self.log = (ROOT / 'logs/llama-server.log').open('a')
        self.process = subprocess.Popen([str(binaries[0]), '-m', str(path),
            '--host', '127.0.0.1', '--port', str(port), '--api-key', self.secret,
            '-c', '8192', '-np', '1', '-ngl', '99', '--jinja', '--no-webui'],
            stdout=self.log, stderr=subprocess.STDOUT, cwd=ROOT)
        try:
            for _ in range(600):
                check()
                if self.process.poll() is not None:
                    raise RuntimeError('llama.cpp 加载失败，详见 logs/llama-server.log')
                try:
                    with self.call('/health', timeout=1) as response:
                        if response.status == 200:
                            self.key = model_id
                            return
                except OSError:
                    pass
                time.sleep(.2)
            raise TimeoutError('llama.cpp 加载超时')
        except BaseException:
            self.stop()
            raise

    def generate(self, req, check, update, cancel):
        # Cancellation also interrupts blocked prefill/network reads by stopping the child.
        finished = threading.Event()
        def watch():
            while not finished.wait(.1):
                if cancel.is_set():
                    if self.process:
                        self.process.terminate()
                    return
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            self.start(req['model'], check)
            template = {'messages': req['messages'], 'add_generation_prompt': True,
                        'chat_template_kwargs': {'enable_thinking': req['thinking']}}
            with self.call('/apply-template', template) as r:
                prompt = json.load(r)['prompt']
            with self.call('/tokenize', {'content': prompt, 'add_special': True}) as r:
                count = len(json.load(r)['tokens'])
            if count + req['max_tokens'] > req['context_limit']:
                raise ValueError(f'输入 {count} + 生成上限 {req["max_tokens"]} 超过上下文预算 {req["context_limit"]}')
            result = {'text': '', 'tokens': 0, 'prompt_tokens': count, 'tps': 0,
                      'peak_gb': None, 'backend': 'llama.cpp', 'finish_reason': None}
            update(result)
            body = dict(template, stream=True, stream_options={'include_usage': True},
                        max_tokens=req['max_tokens'], temperature=req['temperature'], seed=req['seed'],
                        reasoning_format='none')
            started = time.monotonic()
            with self.call('/v1/chat/completions', body) as response:
                for line in response:
                    check()
                    if not line.startswith(b'data: '):
                        continue
                    raw = line[6:].strip()
                    if raw == b'[DONE]':
                        break
                    data = json.loads(raw)
                    if data.get('error'):
                        raise RuntimeError(str(data['error']))
                    for choice in data.get('choices', []):
                        result['text'] += choice.get('delta', {}).get('content') or ''
                        if choice.get('finish_reason'):
                            result['finish_reason'] = choice['finish_reason']
                    usage = data.get('usage') or {}
                    if usage:
                        result['tokens'] = usage.get('completion_tokens', 0)
                        result['prompt_tokens'] = usage.get('prompt_tokens', count)
                        result['tps'] = round(result['tokens'] / max(.001, time.monotonic()-started), 2)
                    update(result)
            check()
        except Exception:
            if cancel.is_set():
                self.stop()
                check()
            raise
        finally:
            finished.set()
            watcher.join(timeout=1)
