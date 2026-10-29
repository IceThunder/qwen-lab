"""Read-only Neuronary recipe import. Only the fixed public API is contacted."""
import json
import math
import re
import urllib.error
import urllib.request
from urllib.parse import urlsplit

API = 'https://mcczbdcymyhzfputyqcr.supabase.co/functions/v1/api/recipes/'
MAX_BYTES = 1024 * 1024


def recipe_id(value):
    if not isinstance(value, str) or len(value) > 500:
        raise ValueError('请输入 Neuronary 单条配方链接或 ID')
    value = value.strip()
    if re.fullmatch(r'[a-zA-Z0-9_-]{4,64}', value):
        return value
    u = urlsplit(value)
    if u.scheme != 'https' or u.netloc != 'neuronary.org' or u.query:
        raise ValueError('仅支持 https://neuronary.org/r/<ID> 链接，或配方 ID')
    match = re.fullmatch(r'/r/([a-zA-Z0-9_-]{4,64})/?', u.path)
    if not match:
        raise ValueError('请复制单条配方的 /r/<ID> 链接，而不是网站首页')
    return match[1]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch_recipe(value):
    rid = recipe_id(value)
    request = urllib.request.Request(API + rid, headers={'Accept': 'application/json', 'User-Agent': 'QwenLab/1.0'})
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=15) as response:
            raw = response.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError('网站配方超过 1 MB，未导入')
        data = json.loads(raw)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ValueError('没有找到该配方，链接可能已失效或不是公开配方') from exc
        raise ValueError(f'Neuronary 返回 HTTP {exc.code}，请稍后重试') from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('读取 Neuronary 失败，请检查网络后重试') from exc
    if not isinstance(data, dict) or data.get('id') != rid:
        raise ValueError('网站返回的配方格式或 ID 不匹配')
    return data


def text(data, key, limit=30000):
    value = data.get(key, '')
    if value is None:
        return ''
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f'配方 {key} 字段格式无效或过长')
    return value


def normalize(data, catalog):
    if not isinstance(data, dict):
        raise ValueError('配方必须是 JSON 对象')
    rid = recipe_id(data.get('id'))
    model = text(data, 'model', 200)
    name = text(data, 'name', 500)
    neurons = data.get('neurons')
    if not isinstance(neurons, list) or not 1 <= len(neurons) <= 32:
        raise ValueError('仅支持含 1–32 个神经元的单条配方')
    groups, seen = [], set()
    for item in neurons:
        if not isinstance(item, dict) or item.get('expert') is not None:
            raise ValueError('当前仅支持普通 MLP 通道，不支持 MoE expert 配方')
        layer, neuron, dose = item.get('layer'), item.get('neuron'), item.get('dose')
        if any(type(v) is not int or v < 0 for v in (layer, neuron)):
            raise ValueError('配方层号和神经元编号必须是非负整数（从 0 编号）')
        if isinstance(dose, bool) or not isinstance(dose, (int, float)) or not math.isfinite(dose) or abs(dose) > 4:
            raise ValueError('配方剂量必须是 −4 到 +4 的有限数值，不会自动截断剂量')
        if (layer, neuron) in seen:
            raise ValueError('配方包含重复通道，请先在来源网站确认')
        seen.add((layer, neuron))
        groups.append(dict(layer=layer, neurons=str(neuron), mode='normalized_add', dose=dose, scale=1, enabled=True))
    # Deliberate allowlist: model family or channel dimensions alone do not prove compatibility.
    compatible = next((m for m in catalog if m['id'] == 'qwen3.5-4b'
        and m.get('supports_steering') is not False and model == 'Qwen/Qwen3.5-4B'), None)
    if compatible and any(g['layer'] >= compatible['layers'] or int(g['neurons']) >= compatible['width'] for g in groups):
        raise ValueError('配方编号超出本地模型层数或通道范围')
    warnings = ['本地使用 Qwen3.5-4B 的 MLX 4-bit 转换权重；未确认网站作者的权重版本及精度，效果不保证一致。',
                '本地按当前完整输入无干预校准、跳过首 token，固定偏移用于全部输入与生成 token；网站未完整规定校准细节。']
    sampling = data.get('sampling') or {}
    if not isinstance(sampling, dict):
        raise ValueError('配方 sampling 必须为对象')
    if len(json.dumps(sampling, ensure_ascii=False)) > 10000:
        raise ValueError('采样设置过长')
    temperature = sampling.get('temperature', 0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or not 0 <= temperature <= 1.5:
        warnings.append('网站 temperature 超出本地支持范围，导入时使用 0；不属于严格复现。')
        temperature = 0
    unsupported = sorted(k for k in sampling if k != 'temperature')
    if unsupported:
        warnings.append('以下网站采样设置仅保留为参考，当前不会应用：' + ', '.join(unsupported))
    if 'temperature' not in sampling:
        warnings.append('网站未提供 temperature，导入使用本地测试默认值 0。')
    examples = data.get('examples') or []
    if not isinstance(examples, list) or len(examples) > 100:
        raise ValueError('示例列表格式无效或过多')
    clean_examples = []
    for ex in examples:
        if not isinstance(ex, dict):
            raise ValueError('示例格式无效')
        prompt = text(ex, 'prompt')
        if not prompt.strip():
            continue
        clean_examples.append({key: text(ex, key) for key in ('prompt', 'output', 'base_output')})
    return dict(id=rid, url=f'https://neuronary.org/r/{rid}', name=name, model=model,
        recipe=text(data, 'recipe'), description=text(data, 'description'), notes=text(data, 'notes'),
        groups=groups, sampling=sampling, temperature=temperature, examples=clean_examples,
        compatible_model=compatible['id'] if compatible else None,
        compatibility='可尝试：原有 4B MLX（量化转换版）' if compatible else '无法导入：本地没有已适配的对应模型；GGUF 不支持这套神经元干预。',
        warnings=warnings if compatible else [],
        local_convention=dict(calibration_skip_first=True, thinking=False, compare=True,
                              method='baseline_prompt_median', scope='all_prompt_and_generated_tokens'))
