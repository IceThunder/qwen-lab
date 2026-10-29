"""Local-only model discovery, validated chat jobs and reversible MLP interventions."""
import copy
import gc
import json
import math
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def discover_models():
    result = []
    for folder in sorted((ROOT / 'models').glob('*')):
        if (folder / 'runtime.json').is_file():
            try:
                from gguf_backend import specification
                meta, _ = specification(folder)
                result.append({'id': folder.name, 'name': meta['name'], 'layers': meta.get('layers'),
                    'width': 0, 'format': meta.get('format', 'GGUF · ' + __import__('model_manager').quant_label(meta['file'])) + ' · llama.cpp', 'adapters': [],
                    'backend': 'llama.cpp', 'supports_steering': False})
            except (ValueError, KeyError, OSError):
                pass
            continue
        config_path = folder / 'weights/config.json'
        if not config_path.is_file() or not list((folder / 'weights').glob('*.safetensors')):
            continue
        try:
            config = json.loads(config_path.read_text())
            text = config.get('text_config', config)
            if config.get('model_type') not in ('qwen3_5', 'qwen3_5_text'):
                continue
            adapters = []
            for adapter in sorted((ROOT / 'adapters' / folder.name).glob('*')):
                if (adapter / 'adapters.safetensors').is_file() and (adapter / 'adapter_config.json').is_file():
                    adapters.append(adapter.name)
            installation = json.loads((folder / 'installation.json').read_text()) if (folder / 'installation.json').exists() else {}
            result.append({'id': folder.name, 'name': installation.get('name', folder.name.replace('qwen', 'Qwen ').replace('-4b', ' · 4B')),
                           'layers': text['num_hidden_layers'], 'width': text['intermediate_size'],
                           'format': f"MLX · {config['quantization']['bits']}-bit" if config.get('quantization', {}).get('bits') else 'MLX',
                           'adapters': adapters})
        except (ValueError, KeyError, OSError):
            continue
    for item in result:
        try:
            last = json.loads((ROOT / 'models' / item['id'] / 'load_test.json').read_text())
            item['last_load_test'] = {'status': last['status'], 'tested_at': last['tested_at'], 'error': last.get('error', '')}
        except (OSError, ValueError, KeyError):
            pass
    return result


def integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f'{name} 必须为 {low}–{high} 的整数')
    return value


def number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} 必须为 {low}–{high} 的有限数值')
    return value


def parse_neurons(value, width):
    """Comma-separated IDs / inclusive ranges, or a list. Deduplicated per group."""
    ids = set()
    if isinstance(value, str):
        if len(value) > 20000:
            raise ValueError('神经元表达式过长')
        for part in value.replace('，', ',').split(','):
            part = part.strip()
            if not part:
                continue
            try:
                if '-' in part:
                    a, b = (int(x.strip()) for x in part.split('-'))
                    if not 0 <= a <= b < width:
                        raise ValueError()
                    ids.update(range(a, b + 1))
                else:
                    i = int(part)
                    if not 0 <= i < width:
                        raise ValueError()
                    ids.add(i)
            except ValueError:
                raise ValueError(f'无效的神经元编号：{part}；可用范围 0–{width-1}')
    elif isinstance(value, list):
        for i in value:
            ids.add(integer(i, '神经元', 0, width - 1))
    else:
        raise ValueError('神经元需为编号列表，如 0,1,10-20')
    if not ids:
        raise ValueError('每个已启用的干预组至少选择一个神经元')
    return sorted(ids)


def validate_request(body, catalog=None):
    if not isinstance(body, dict):
        raise ValueError('请求必须为 JSON 对象')
    catalog = discover_models() if catalog is None else catalog
    spec = next((m for m in catalog if m['id'] == body.get('model')), None)
    if spec is None:
        raise ValueError('请选择已安装并受支持的模型')
    adapter = body.get('adapter', '')
    if not isinstance(adapter, str) or (adapter and adapter not in spec['adapters']):
        raise ValueError('该模型下不存在所选适配器')
    messages = body.get('messages')
    if not isinstance(messages, list) or not 1 <= len(messages) <= 80:
        raise ValueError('对话必须包含 1–80 条消息')
    clean = []
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError('消息格式无效')
        role, content = message.get('role'), message.get('content')
        if role not in ('system', 'user', 'assistant') or (role == 'system' and i != 0):
            raise ValueError('系统提示词只能出现在第一条消息')
        if not isinstance(content, str) or not content.strip() or len(content) > 30000:
            raise ValueError('每条消息需包含 1–30000 个字符')
        clean.append({'role': role, 'content': content})
    turns = clean[1:] if clean[0]['role'] == 'system' else clean
    if not turns or any(m['role'] != ('user' if i % 2 == 0 else 'assistant') for i, m in enumerate(turns)) or turns[-1]['role'] != 'user':
        raise ValueError('历史消息必须按用户、助手交替排列，并以用户问题结束')
    if sum(len(m['content']) for m in clean) > 100000:
        raise ValueError('对话过长，请新建会话或缩短历史')
    groups = body.get('groups', [])
    if not isinstance(groups, list) or len(groups) > 32:
        raise ValueError('最多配置 32 个干预组')
    if spec.get('supports_steering') is False and (groups or body.get('compare')):
        raise ValueError('GGUF 模型目前仅支持对话，不支持神经元干预或对照模式')
    normalized, occupied = [], set()
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get('enabled', True), bool):
            raise ValueError('干预组格式无效')
        if not group.get('enabled', True):
            continue
        layer = integer(group.get('layer'), '层号', 0, spec['layers'] - 1)
        mode = group.get('mode', 'scale')
        if mode not in ('scale', 'normalized_add'):
            raise ValueError('不支持的干预模式')
        value = number(group.get('dose') if mode == 'normalized_add' else group.get('scale'),
                       '归一化剂量' if mode == 'normalized_add' else '缩放倍率',
                       -4 if mode == 'normalized_add' else 0, 4)
        neurons = parse_neurons(group.get('neurons'), spec['width'])
        for neuron in neurons:
            if (layer, neuron) in occupied:
                raise ValueError(f'第 {layer} 层神经元 {neuron} 在多个启用组中重复，请合并或禁用其中一组')
            occupied.add((layer, neuron))
        normalized.append({'layer': layer, 'neurons': neurons, 'mode': mode,
                           'dose' if mode == 'normalized_add' else 'scale': value})
    if sum(len(g['neurons']) for g in normalized if g['mode'] == 'normalized_add') > 128:
        raise ValueError('归一化加法每次最多选择 128 个通道')
    for key in ('compare', 'thinking', 'calibration_skip_first'):
        if not isinstance(body.get(key, False), bool):
            raise ValueError(f'{key} 必须是布尔值')
    return {'model': spec['id'], 'adapter': adapter, 'messages': clean,
            'groups': normalized, 'compare': body.get('compare', False),
            'thinking': body.get('thinking', False),
            'calibration_skip_first': body.get('calibration_skip_first', True),
            'max_tokens': integer(body.get('max_tokens', 256), '生成长度', 1, 4096),
            'temperature': number(body.get('temperature', 0), '温度', 0, 1.5),
            'seed': integer(body.get('seed', 42), '随机种子', 0, 2147483647),
            'context_limit': integer(body.get('context_limit', 2048), '上下文预算', 128, 8192)}


@contextmanager
def apply_groups(model, groups, width, audit=None):
    import mlx.core as mx
    import mlx.nn as nn

    class AffineProjection(nn.Module):
        def __init__(self, original, mask, offsets, layer, selected):
            super().__init__()
            self.original = original
            self.mask = mx.array(mask)
            self.offsets = mx.array(offsets, dtype=mx.float32)
            self.additive = any(v != 0 for v in offsets)
            self.layer = layer
            self.selected = selected
            self.observed = False

        def __call__(self, x):
            y = x * self.mask.astype(x.dtype)
            if self.additive:
                y = (y.astype(mx.float32) + self.offsets).astype(x.dtype)
            if audit is not None and self.selected and not self.observed:
                before = x[0, 0, self.selected].astype(mx.float32).tolist()
                after = y[0, 0, self.selected].astype(mx.float32).tolist()
                for n, a, b in zip(self.selected, before, after):
                    audit.append({'layer': self.layer, 'neuron': n, 'before': a,
                                  'after': b, 'actual_delta': b-a})
                self.observed = True
            return self.original(y)

    masks, offsets, selected = {}, {}, {}
    for group in groups:
        layer = group['layer']
        mask = masks.setdefault(layer, [1.0] * width)
        shift = offsets.setdefault(layer, [0.0] * width)
        if group.get('mode', 'scale') == 'normalized_add':
            values = group.get('offsets')
            if values is None or len(values) != len(group['neurons']):
                raise ValueError('归一化加法尚未校准，不能直接运行')
            for n, delta in zip(group['neurons'], values):
                shift[n] = delta
                selected.setdefault(layer, []).append(n)
        else:
            for n in group['neurons']:
                mask[n] = group['scale']
    originals = []
    try:
        for layer, mask in masks.items():
            mlp = model.layers[layer].mlp
            originals.append((mlp, mlp.down_proj))
            mlp.down_proj = AffineProjection(mlp.down_proj, mask, offsets[layer], layer, selected.get(layer, []))
        yield
    finally:
        for mlp, original in reversed(originals):
            mlp.down_proj = original


class Cancelled(Exception):
    pass


class JobManager:
    """All MLX work stays on one persistent worker. HTTP threads only read state."""
    def __init__(self):
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='qwen-mlx')
        self.jobs = {}
        self.active = None
        self.model = self.tokenizer = self.loaded_key = None
        from gguf_backend import GGUFEngine
        self.gguf = GGUFEngine()

    def submit(self, request):
        with self.lock:
            if self.active:
                raise RuntimeError('已有实验运行中，请先等待完成或停止当前实验')
            job_id = uuid.uuid4().hex
            self.jobs[job_id] = {'id': job_id, 'status': 'queued', 'phase': '等待加载模型',
                                 'results': {}, 'error': '', 'cancel': threading.Event(),
                                 'created': time.time(), 'request': request}
            self.active = job_id
            # Bound RAM use; browser keeps its own conversation history.
            if len(self.jobs) > 30:
                del self.jobs[next(iter(self.jobs))]
            self.pool.submit(self._run, job_id)
            return job_id

    def snapshot(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return copy.deepcopy({k: v for k, v in self.jobs[job_id].items() if k not in ('cancel', 'request')})

    def cancel(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            self.jobs[job_id]['cancel'].set()

    def update(self, job_id, **values):
        with self.lock:
            self.jobs[job_id].update(values)

    def _run(self, job_id):
        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
        job = self.jobs[job_id]
        req, event = job['request'], job['cancel']
        def check(*_):
            if event.is_set():
                raise Cancelled()
        try:
            self.update(job_id, status='running', phase='加载模型')
            check()
            if (ROOT / 'models' / req['model'] / 'runtime.json').is_file():
                self.model = self.tokenizer = self.loaded_key = None
                gc.collect()
                mx.clear_cache()
                def publish(result):
                    self.update(job_id, phase='llama.cpp 当前配置生成中',
                        results={'intervened': copy.deepcopy(result)})
                self.gguf.generate(req, check, publish, event)
                if req.get('load_test') and not job['results'].get('intervened', {}).get('text', '').strip():
                    raise ValueError('模型已加载但短文本测试没有返回内容；请检查聊天模板与模型类型')
                self.update(job_id, status='done', phase='完成')
                return
            self.gguf.stop()
            key = (req['model'], req['adapter'])
            if key != self.loaded_key:
                self.model = self.tokenizer = self.loaded_key = None
                gc.collect()
                mx.clear_cache()
                path = ROOT / 'models' / req['model'] / 'weights'
                adapter = str(ROOT / 'adapters' / req['model'] / req['adapter']) if req['adapter'] else None
                self.model, self.tokenizer = load(str(path), adapter_path=adapter)
                self.loaded_key = key
            check()
            tokens = self.tokenizer.apply_chat_template(req['messages'], tokenize=True,
                return_dict=False, add_generation_prompt=True, enable_thinking=req['thinking'])
            if len(tokens) + req['max_tokens'] > req['context_limit']:
                raise ValueError(f'输入 {len(tokens)} + 生成上限 {req["max_tokens"]} 超过上下文预算 {req["context_limit"]}；请缩短历史、降低生成长度或提高预算（最多 8192）')
            config = json.loads((ROOT / 'models' / req['model'] / 'weights/config.json').read_text())
            width = config.get('text_config', config)['intermediate_size']
            from steering import calibrate
            if any(g.get('mode') == 'normalized_add' for g in req['groups']):
                self.update(job_id, phase='无干预预运行：校准残差尺度')
            resolved, calibration = calibrate(self.model, tokens, req['groups'],
                skip_first=req['calibration_skip_first'], check=check)
            self.update(job_id, calibration=calibration)
            phases = [('baseline', [])] if req['compare'] else []
            phases.append(('intervened', resolved))
            results = {}
            for name, groups in phases:
                check()
                mx.random.seed(req['seed'])
                mx.reset_peak_memory()
                self.update(job_id, phase='原始对照生成中' if name == 'baseline' else '当前配置生成中')
                result = {'text': '', 'tokens': 0, 'prompt_tokens': len(tokens), 'tps': 0,
                          'peak_gb': 0, 'finish_reason': None}
                results[name] = result
                self.update(job_id, results=copy.deepcopy(results))
                audit = []
                with apply_groups(self.model, groups, width, audit=audit):
                    generator = stream_generate(self.model, self.tokenizer, prompt=tokens,
                        max_tokens=req['max_tokens'], sampler=make_sampler(temp=req['temperature']),
                        prefill_step_size=128, prompt_progress_callback=check)
                    try:
                        for response in generator:
                            check()
                            result.update(text=result['text'] + response.text,
                                tokens=response.generation_tokens, tps=round(response.generation_tps, 2),
                                peak_gb=round(response.peak_memory, 3), finish_reason=response.finish_reason)
                            if name == 'intervened' and calibration is not None:
                                self.update(job_id, activation_audit=copy.deepcopy(audit))
                            self.update(job_id, results=copy.deepcopy(results))
                    finally:
                        generator.close()
            check()
            if req.get('load_test') and not results.get('intervened', {}).get('text', '').strip():
                raise ValueError('模型已加载但短文本测试没有返回内容；请检查聊天模板与模型类型')
            self.update(job_id, status='done', phase='完成')
        except Cancelled:
            self.update(job_id, status='cancelled', phase='已停止；不完整回答不会写入后续上下文')
        except Exception as exc:
            self.update(job_id, status='error', phase='运行失败', error=str(exc))
        finally:
            mx.clear_cache()
            if req.get('load_test'):
                from model_manager import atomic_json
                snapshot = self.snapshot(job_id)
                try:
                    atomic_json(ROOT / 'models' / req['model'] / 'load_test.json', {
                        'status': snapshot['status'], 'error': snapshot.get('error', ''),
                        'results': snapshot['results'], 'tested_at': time.time(),
                        'note': '仅验证无干预短文本推理；不代表长上下文、神经元干预或微调已验证'})
                except OSError:
                    self.update(job_id, phase=snapshot['phase'] + '；测试记录未能写入磁盘')
            with self.lock:
                self.active = None
