"""Residual-normalized additive steering; baseline prompt calibration, zero-based layers."""
import math
import statistics
from contextlib import contextmanager
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache


def column_norms(projection, neurons):
    """L2 norms of effective output columns, dequantized before indexing.

    MLX linear weights are [output, input]. For LoRA, include its effective
    low-rank update, in eval mode, before taking the norm.
    """
    base = projection.linear if hasattr(projection, 'lora_a') else projection
    if isinstance(base, nn.QuantizedLinear):
        weight = mx.dequantize(base.weight, base.scales, base.biases,
                              group_size=base.group_size, bits=base.bits, mode=base.mode)
    elif isinstance(base, nn.Linear):
        weight = base.weight
    else:
        raise ValueError(f'不支持计算该 down_proj 的列范数：{type(base).__name__}')
    columns = weight[:, neurons].astype(mx.float32)
    if hasattr(projection, 'lora_a'):
        columns = columns + projection.scale * (
            projection.lora_a[neurons].astype(mx.float32) @ projection.lora_b.astype(mx.float32)
        ).T
    norms = mx.sqrt(mx.sum(columns * columns, axis=0))
    values = norms.tolist()
    if any(not math.isfinite(n) or n <= 1e-12 for n in values):
        raise ValueError('所选权重列范数为零或非有限值，无法归一化')
    return values


@contextmanager
def capture_residuals(model, targets, collected, check):
    # model.layers is a COPY of pipeline_layers in mlx-lm: replace the actual
    # list on the text backbone, otherwise no capture would run.
    layers = model.model.layers
    originals = []

    class CaptureLayer(nn.Module):
        def __init__(self, original, index):
            super().__init__()
            self.original = original
            self.index = index
            self.is_linear = original.is_linear

        def __call__(self, *args, **kwargs):
            check()
            output = self.original(*args, **kwargs)
            # Post-residual-addition decoder output, BEFORE final backbone norm.
            norms = mx.sqrt(mx.sum(output[0].astype(mx.float32) ** 2, axis=-1))
            collected[self.index].extend(norms.tolist())
            return output

    try:
        for index in sorted(targets):
            original = layers[index]
            originals.append((index, original))
            layers[index] = CaptureLayer(original, index)
        yield
    finally:
        for index, original in reversed(originals):
            layers[index] = original


def calibrate(model, tokens, groups, skip_first=True, check=lambda: None, chunk_size=128):
    additive = [g for g in groups if g.get('mode', 'scale') == 'normalized_add']
    if not additive:
        return groups, None
    skip = 1 if skip_first else 0
    if len(tokens) <= skip:
        raise ValueError('跳过首 token 后没有可用于校准的 token')
    targets = {g['layer'] for g in additive}
    collected = {index: [] for index in targets}
    cache = make_prompt_cache(model)
    try:
        with capture_residuals(model, targets, collected, check):
            for start in range(0, len(tokens), chunk_size):
                check()
                # Backbone only: no full-sequence vocabulary logits allocation.
                hidden = model.model(mx.array([tokens[start:start + chunk_size]]), cache=cache)
                mx.eval(hidden)
                mx.eval([c.state for c in cache])
        medians = {}
        for index, values in collected.items():
            if len(values) != len(tokens):
                raise ValueError(f'第 {index} 层校准 token 数不匹配')
            medians[index] = statistics.median(values[skip:])
            if not math.isfinite(medians[index]) or medians[index] <= 0:
                raise ValueError(f'第 {index} 层残差尺度无效')
    finally:
        del cache
        mx.clear_cache()
    resolved, rows = [], []
    for group in groups:
        check()
        item = dict(group)
        if group.get('mode', 'scale') == 'normalized_add':
            norms = column_norms(model.layers[group['layer']].mlp.down_proj, group['neurons'])
            residual = medians[group['layer']]
            offsets = [group['dose'] * residual / n for n in norms]
            if any(not math.isfinite(d) or abs(d) > 1e20 for d in offsets):
                raise ValueError('归一化注入量溢出')
            item['offsets'] = offsets
            for neuron, norm, delta in zip(group['neurons'], norms, offsets):
                rows.append({'layer': group['layer'], 'neuron': neuron, 'dose': group['dose'],
                             'residual_median': residual, 'column_norm': norm, 'delta': delta})
        resolved.append(item)
    return resolved, {'method': 'baseline_prompt_median', 'skip_first_token': skip_first,
                      'calibration_tokens': len(tokens) - skip, 'prompt_tokens': len(tokens),
                      'scope': 'all_prompt_and_generated_tokens',
                      'reference': 'same_model_and_adapter_without_any_activation_intervention',
                      'median_rule': 'average_middle_two_for_even_count', 'rows': rows}
