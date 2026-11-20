"""Head orientation followed by context actions, with explicit suspected fallbacks."""
import copy
import hashlib
import time

from . import prompts, vlm

VERSION = 'head_context_strategy1_v2'
SUSPECTED = {'抬头': '疑似抬头', '低头': '疑似低头'}


def context_profile(config: dict) -> dict:
    value = copy.deepcopy(config)
    value.pop('head_only_short', None)
    for action in value['actions']:
        if action['name'] in SUSPECTED:
            action['name'] = SUSPECTED[action['name']]
            action['priority'] = 0
        else:
            action['priority'] = max(1, action['priority'])
    value['instructions'] += ('\n本次为第二阶段扩展 ROI 分析。普通抬头、低头仅输出疑似抬头、疑似低头。'
                              '先检查有可见证据的手机、平板、写字、看书、趴伏等启用具体动作，'
                              '没有具体动作证据时才选疑似姿态；无依据时仍输出无法判断。')
    # Leave room for the pipeline instruction even at the editable limit.
    if len(value['instructions']) > 4000:
        raise ValueError('完整动作配置的附加说明过长，请缩短以容纳两阶段说明')
    return prompts.normalize(value)


def cache_key(head_path, context_path, model, quantization, head_profile, context, experiment, batch_size=1):
    parts = [VERSION,
             vlm.cache_key(head_path, model, quantization, head_profile, experiment, batch_size),
             vlm.cache_key(context_path, model, quantization, context, experiment, batch_size)]
    return hashlib.sha256('|'.join(parts).encode()).hexdigest()


def merge(head: dict, context: dict) -> dict:
    if head['state'] == '抬头':
        chosen, decision = head, 'head_confirmed'
    elif context['state'] not in {*SUSPECTED.values(), prompts.UNKNOWN}:
        chosen, decision = context, 'context_action'
    elif head['state'] in SUSPECTED:
        chosen, decision = head, 'head_confirmed'
    elif context['state'] in SUSPECTED.values():
        chosen, decision = context, 'context_suspected'
    else:
        chosen = {**context, 'state': '疑似低头',
                  'evidence': '两阶段均无法判断；按策略兜底为疑似低头，并非已确认低头。',
                  'uncertainty': context.get('uncertainty') or '无足够可见证据，规则兜底'}
        decision = 'fallback'
    return {**chosen, 'decision': decision, 'fallback': decision == 'fallback',
            'stages': {'head': head, 'context': context}, 'pipeline_version': VERSION}


def combine(head: dict, second: dict) -> dict:
    result = merge(head, second)
    # Record combined costs; do not report just the winning stage's generation.
    metrics = [head.get('metrics', {}), second.get('metrics', {})]
    peaks = [m['peak_allocated_mib'] for m in metrics if m.get('peak_allocated_mib') is not None]
    result['metrics'] = {
        **second.get('metrics', {}),
        'generation_seconds': sum(m.get('generation_seconds', 0) for m in metrics),
        'model_load_seconds': sum(m.get('model_load_seconds', 0) for m in metrics),
        'generated_tokens': sum(m.get('generated_tokens', 0) for m in metrics),
        'cold_load': any(m.get('cold_load', False) for m in metrics),
        'peak_allocated_mib': max(peaks) if peaks else None,
    }
    return result


def run_batch(entries, model, quantization, head_profile, context, experiment,
              progress, publish, cancelled=lambda: False, *, batch_size=1):
    """Publish every Head result before any expanded-context model call."""
    heads = {}
    total = len(entries)
    for phase in ('head', 'context'):
        profile = head_profile if phase == 'head' else context
        for start in range(0, total, batch_size):
            if cancelled():
                return
            progress(phase, start, total)
            end = min(start + batch_size, total)
            indices = [index for index in range(start, end) if phase == 'head' or index in heads]
            if indices:
                paths = [entries[index][f'{phase}_path'] for index in indices]
                started = time.perf_counter()
                try:
                    results = vlm.analyze_group(paths, model, quantization, profile, experiment)
                except Exception as error:
                    results = [error] * len(indices)
                elapsed = (time.perf_counter() - started) / len(indices)
                for index, result in zip(indices, results, strict=True):
                    entry = entries[index]
                    if isinstance(result, Exception):
                        publish(entry, phase, heads.get(index), result, elapsed)
                        if isinstance(result, RuntimeError):
                            raise result
                        continue
                    result = copy.deepcopy(result)
                    result.update(input_file=entry[f'{phase}_file'], region_bbox=entry[f'{phase}_bbox'])
                    if phase == 'head':
                        heads[index] = result
                    else:
                        result = combine(heads[index], result)
                    publish(entry, phase, result, None, elapsed)
            progress(phase, end, total)
