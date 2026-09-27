"""Recorded teacher traces -> causal SCHNITZELJAGD support/query episodes.

Prefix context compression and cross-experience transfer are separate protocols.
Targets are complete messages; no tools or remote dataset Python are executed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json

from .data import Episode, Source, save_episodes
from .text import render_prompt

CATALOG = {
    'hermes': dict(path='NousResearch/hermes-function-calling-v1', config='func_calling',
                   split='train', adapter='hermes', license='apache-2.0'),
    'ultrachat': dict(path='HuggingFaceH4/ultrachat_200k', config='default',
                      split='train_sft', adapter='ultrachat', license='mit'),
    'swe_smith': dict(path='SWE-bench/SWE-smith-trajectories', config='default',
                      split='tool', adapter='swe_smith', license='mit', success_only=True),
    'openhands': dict(path='nebius/SWE-rebench-openhands-trajectories', config='default',
                      split='train', adapter='openhands', license='cc-by-4.0', success_only=True),
    'xlam': dict(path='Salesforce/xlam-function-calling-60k', config=None,
                 split='train', adapter='xlam', license='cc-by-4.0', gated=True),
}


def digest(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def parse_json(value):
    # SWE-smith includes JSON strings; tolerate one additional serialization layer.
    for _ in range(2):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            break
    return value


def text_content(value) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    if isinstance(value, list):
        parts = []
        for block in value:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get('type') in {'text', 'input_text', 'output_text'}:
                parts.append(str(block.get('text', '')))
            elif isinstance(block, dict) and block.get('type') == 'tool_use':
                call = {'name': block.get('name'), 'arguments': block.get('input', {})}
                if block.get('id'):
                    call['id'] = block['id']
                parts.append('<tool_call>' + json.dumps(call, sort_keys=True) + '</tool_call>')
            elif isinstance(block, dict) and block.get('type') == 'tool_result':
                prefix = f"tool_call_id={block['tool_use_id']}\n" if block.get('tool_use_id') else ''
                parts.append(prefix + text_content(block.get('content')))
            else:
                raise ValueError('Unsupported nontext content block')
        return '\n'.join(parts)
    raise ValueError('Unsupported content type')


def normalize_message(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError('Unsupported message object')
    role = raw.get('role', raw.get('from'))
    role = {'human': 'user', 'gpt': 'assistant', 'model': 'assistant', 'agent': 'assistant',
            'function': 'tool', 'observation': 'tool', 'environment': 'tool'}.get(role, role)
    if role not in {'user', 'assistant', 'system', 'tool'}:
        raise ValueError(f'Unsupported role {role!r}')
    content = text_content(raw.get('content', raw.get('value')))
    calls = parse_json(raw.get('tool_calls'))
    if calls:
        if not isinstance(calls, list):
            raise ValueError('tool_calls must be a list')
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get('function', call), dict):
                raise ValueError('Unsupported tool call object')
            function = call.get('function', call)
            value = {'name': function.get('name'), 'arguments': parse_json(function.get('arguments', {}))}
            if call.get('id'):
                value['id'] = call['id']
            content += ('\n' if content else '') + '<tool_call>' + json.dumps(value, sort_keys=True, ensure_ascii=False) + '</tool_call>'
    if role == 'tool' and (raw.get('name') or raw.get('tool_call_id')):
        content = json.dumps({k: raw[k] for k in ('name', 'tool_call_id') if raw.get(k)}, sort_keys=True) + '\n' + content
    # Whitespace is meaningful in code; never strip source/target content.
    return {'role': role, 'content': content}


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    instance_id: str
    split_group: str
    transfer_group: str | None
    dataset: str
    revision: str
    license: str
    teacher: str
    messages: tuple[dict, ...]
    content_hash: str


def normalize_trajectory(row: dict, spec: dict) -> Trajectory:
    adapter = spec['adapter']
    if spec.get('success_only') and row.get('resolved') not in (True, 1, 'true', '1'):
        raise ValueError('filtered_unsuccessful')
    tools = parse_json(row.get('tools'))
    if adapter == 'xlam':
        raw = [dict(role='system', content='Available tools:\n' + json.dumps(tools or [], sort_keys=True)),
               dict(role='user', content=row['query']),
               dict(role='assistant', content=json.dumps(parse_json(row['answers']), sort_keys=True))]
    else:
        field = {'hermes': 'conversations', 'ultrachat': 'messages', 'messages': 'messages',
                 'swe_smith': 'messages', 'openhands': 'trajectory'}[adapter]
        raw = parse_json(row.get(field))
    if not isinstance(raw, list):
        raise ValueError('missing_message_list')
    messages = [normalize_message(m) for m in raw]
    if tools and not any('Available tools:' in m['content'] or '<tools>' in m['content']
                         for m in messages if m['role'] == 'system'):
        messages.insert(0, dict(role='system', content='Available tools:\n' + json.dumps(tools, sort_keys=True)))
    if not any(m['role'] == 'user' for m in messages) or not any(m['role'] == 'assistant' for m in messages):
        raise ValueError('missing_user_or_assistant')
    content_hash = digest(messages)
    upstream = str(row.get('trajectory_id', row.get('traj_id', row.get('id', row.get('prompt_id', content_hash)))))
    instance = str(row.get('instance_id') or row.get('prompt_id') or upstream)
    transfer = None
    if adapter in {'swe_smith', 'openhands'}:
        repo = row.get('repo')
        if not repo and adapter == 'swe_smith':
            repo = instance.split('.')[0].replace('__', '/')
        if not repo or '/' not in repo:
            raise ValueError('missing_repository_identity')
        group = transfer = 'repo:' + repo
    elif adapter in {'hermes', 'xlam'} and tools:
        group = transfer = 'tools:' + digest(tools)
    else:
        first_user = next(m['content'] for m in messages if m['role'] == 'user')
        group = 'prompt:' + digest(' '.join(first_user.lower().split()))
    return Trajectory(digest([spec['path'], spec['revision'], upstream, content_hash])[:32], instance,
                      group, transfer, spec['path'], spec['revision'], spec.get('license', 'unspecified'),
                      str(row.get('model') or row.get('teacher') or 'not_recorded'), tuple(messages), content_hash)


def split_for(group: str, seed: int, validation_fraction: float) -> str:
    return 'validation' if int(digest([seed, group])[:16], 16) / 2**64 < validation_fraction else 'train'


def chunk_message(tokenizer, trajectory: Trajectory, index: int, max_tokens: int, *, created_at=None) -> list[Source]:
    message = trajectory.messages[index]
    header = f"Recorded {message['role']} context:\n"
    available = max_tokens - len(tokenizer.encode(header, add_special_tokens=True)) - 8
    if available < 16:
        raise ValueError('Source limit too small for framing')
    content, chunks, offset = message['content'], [], 0
    while offset < len(content):
        # Split actual characters, avoiding lossy partial UTF-8 decoding.
        low, high, take = 1, min(len(content) - offset, available * 8), 0
        while low <= high:
            middle = (low + high) // 2
            if len(tokenizer.encode(header + content[offset:offset + middle], add_special_tokens=True)) <= max_tokens:
                take, low = middle, middle + 1
            else:
                high = middle - 1
        if not take:
            raise ValueError('Source limit cannot hold a complete character')
        text = header + content[offset:offset + take]
        created = index + 1 if created_at is None else created_at
        rid = digest([trajectory.trajectory_id, index, offset, text, created])[:32]
        chunks.append(Source(rid, text, created, 'trajectory'))
        offset += take
    return chunks


def recent_query(tokenizer, trajectory, target, cfg):
    first = next(m['content'] for m in trajectory.messages[:target] if m['role'] == 'user')
    first_ids = tokenizer.encode(first, add_special_tokens=False)
    first = tokenizer.decode(first_ids[:cfg.get('task_tokens', 160)], skip_special_tokens=False)
    start = max(0, target - cfg.get('recent_messages', 1))
    recent = '\n'.join(f"{m['role']}: {m['content']}" for m in trajectory.messages[start:target])
    ids = tokenizer.encode(recent, add_special_tokens=False)
    if len(ids) > cfg.get('recent_tokens', 512):
        recent = '[Earlier visible context cropped.]\n' + tokenizer.decode(ids[-cfg.get('recent_tokens', 512):], skip_special_tokens=False)
    return ('Continue this recorded conversation. Earlier context or related experiences may be supplied by SCHNITZELJAGD. '
            'Return only the next assistant message. Preserve JSON tool calls inside <tool_call> tags when applicable.\n'
            'Original task (possibly excerpted):\n' + first + '\nRecent visible messages:\n' + recent)


def _fit_episode(tokenizer, trajectory, target, supports, query_time, cfg, protocol, turns, producers, stats):
    answer = trajectory.messages[target]['content']
    tokens = tokenizer.encode(answer, add_special_tokens=False)
    if not answer.strip() or len(tokens) + int(tokenizer.eos_token_id is not None) > cfg['max_target_tokens']:
        stats['skipped_long_or_empty_target'] += 1
        return None
    query = recent_query(tokenizer, trajectory, target, cfg)
    if len(tokenizer.encode(render_prompt(tokenizer, query), add_special_tokens=False)) > cfg['max_prompt_tokens']:
        stats['skipped_recent_prompt_overflow'] += 1
        return None
    supports = list(supports[-cfg.get('max_supports', 4):])
    while supports and len(tokenizer.encode(render_prompt(tokenizer, query, '\n'.join(s.text for s in supports)),
                                            add_special_tokens=False)) > cfg['max_prompt_tokens']:
        supports.pop(0)
        stats['support_chunks_removed_for_text_control_budget'] += 1
    if not supports:
        stats['skipped_without_memory'] += 1
        return None
    if any(s.created_at >= query_time for s in supports):
        raise AssertionError('Preparation introduced future context')
    ids = tuple(s.record_id for s in supports)
    provenance = dict(dataset=trajectory.dataset, revision=trajectory.revision, license=trajectory.license,
                      teacher=trajectory.teacher, trajectory_id=trajectory.trajectory_id, instance_id=trajectory.instance_id,
                      split_group=trajectory.split_group, target_index=target, target_hash=digest(answer),
                      target_complete=True, protocol=protocol, content_hash=trajectory.content_hash,
                      source_turns={rid: turns[rid] for rid in ids},
                      source_trajectories=sorted({producers[rid] for rid in ids}),
                      ordering='within-conversation' if protocol == 'prefix' else 'experimental ordering; not original timestamps')
    return Episode(digest([trajectory.trajectory_id, target, protocol])[:32], trajectory.trajectory_id,
                   tuple(supports), query, answer, ids, False, 0, 0, query_time,
                   f'trajectory/{protocol}', (), (), 'provided_context', provenance)


def trajectory_episodes(trajectories, tokenizer, cfg, *, protocol='prefix'):
    if protocol not in {'prefix', 'cross_experience'}:
        raise ValueError('Invalid trajectory protocol')
    for key in ('max_source_tokens', 'max_prompt_tokens', 'max_target_tokens', 'max_supports',
                'max_targets_per_trajectory', 'recent_tokens', 'task_tokens', 'support_experiences'):
        if cfg.get(key, 1) < 1:
            raise ValueError(f'Invalid positive budget: {key}')
    if cfg.get('recent_messages', 1) < 0:
        raise ValueError('recent_messages must be nonnegative')
    stats, episodes, previous = Counter(), [], defaultdict(list)
    for rank, trajectory in enumerate(sorted(trajectories, key=lambda t: t.trajectory_id)):
        cache = {}

        def chunks(producer, index, created=None):
            key = (producer.trajectory_id, index, created)
            if key not in cache:
                cache[key] = chunk_message(tokenizer, producer, index, cfg['max_source_tokens'], created_at=created)
            return cache[key]

        targets = [i for i, m in enumerate(trajectory.messages) if m['role'] == 'assistant'
                   and any(p['role'] == 'user' for p in trajectory.messages[:i])]
        cap = cfg.get('max_targets_per_trajectory', 4)
        if len(targets) > cap:
            targets = [targets[i * (len(targets) - 1) // max(1, cap - 1)] for i in range(cap)]
        for target in targets:
            supports, turns, producers = [], {}, {}
            if protocol == 'prefix':
                for index in range(max(0, target - cfg.get('recent_messages', 1))):
                    for source in chunks(trajectory, index):
                        supports.append(source)
                        turns[source.record_id], producers[source.record_id] = index, trajectory.trajectory_id
                query_time = target + 1
            else:
                priors = [p for p in previous[trajectory.transfer_group] if p[1].instance_id != trajectory.instance_id] if trajectory.transfer_group else []
                per_prior = []
                for prior_rank, prior in priors[-cfg.get('support_experiences', 2):]:
                    indices = sorted({next(i for i, m in enumerate(prior.messages) if m['role'] == 'user'),
                                      *range(max(0, len(prior.messages) - 3), len(prior.messages))})
                    group = []
                    for index in indices:
                        for source in chunks(prior, index, prior_rank + 1):
                            group.append(source)
                            turns[source.record_id], producers[source.record_id] = index, prior.trajectory_id
                    per_prior.append(group)
                selected, limit = [], cfg.get('max_supports', 4)
                for depth in range(limit):
                    for group in reversed(per_prior):
                        if len(selected) < limit and len(group) > depth:
                            selected.append(group[-1 - depth])
                supports = list(reversed(selected))
                query_time = rank + 1
            e = _fit_episode(tokenizer, trajectory, target, supports, query_time, cfg, protocol, turns, producers, stats)
            if e:
                episodes.append(e)
                stats['episodes'] += 1
                stats['complete_target_tokens'] += len(tokenizer.encode(e.answer, add_special_tokens=False))
        if trajectory.transfer_group:
            previous[trajectory.transfer_group].append((rank, trajectory))
    return episodes, stats


def pinned_spec(request):
    spec = CATALOG.get(request.get('name'), {}) | request
    if spec.get('local_file'):
        return spec | {'path': spec.get('path', 'local-fixture'), 'revision': file_sha256(spec['local_file'])}
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(spec['path'], revision=spec.get('revision', 'main'))
    if not info.sha:
        raise RuntimeError('Cannot resolve dataset revision')
    return spec | {'revision': info.sha}


def source_rows(spec):
    if spec.get('local_file'):
        with open(spec['local_file'], encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError('Install schnitz[data] or use the Spark image') from e
    source = load_dataset(spec['path'], name=spec.get('config'), revision=spec['revision'],
                          split=spec.get('split', 'train'), streaming=True)
    if spec.get('shuffle_buffer', 0):
        source = source.shuffle(seed=spec.get('shuffle_seed', 17), buffer_size=spec['shuffle_buffer'])
    yield from source


def prepare_trajectories(requests, tokenizer, cfg, output, *, protocol='prefix', seed=17, validation_fraction=.1):
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be in (0,1)')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    splits, hashes, sources = {'train': [], 'validation': []}, set(), []
    with (output / 'normalized.jsonl').open('w', encoding='utf-8') as audit:
        for request in requests:
            spec = pinned_spec(request)
            spec.setdefault('shuffle_seed', seed)
            stats = Counter()
            limit = spec.get('max_rows', 2000)
            if limit < 1:
                raise ValueError('max_rows must be positive')
            skip_rows = spec.get('skip_rows', 0)
            if type(skip_rows) is not int or skip_rows < 0:
                raise ValueError('skip_rows must be a nonnegative integer')
            excluded_groups = spec.get('excluded_split_groups', [])
            if (not isinstance(excluded_groups, list)
                    or any(not isinstance(group, str) or not group for group in excluded_groups)
                    or len(excluded_groups) != len(set(excluded_groups))):
                raise ValueError('excluded_split_groups must be unique nonempty strings')
            excluded_groups = set(excluded_groups)
            iterator = iter(source_rows(spec))
            for _ in range(skip_rows):
                try:
                    next(iterator)
                except StopIteration:
                    break
                stats['rows_skipped'] += 1
            for _ in range(limit):
                try:
                    row = next(iterator)
                except StopIteration:
                    break
                stats['rows_scanned'] += 1
                try:
                    t = normalize_trajectory(row, spec)
                except (ValueError, KeyError, TypeError) as e:
                    stats['filtered:' + str(e)[:100]] += 1
                    continue
                if t.split_group in excluded_groups:
                    stats['excluded_prior_group'] += 1
                    continue
                if t.content_hash in hashes:
                    stats['exact_duplicate_trajectories'] += 1
                    continue
                hashes.add(t.content_hash)
                if sum(len(m['content']) for m in t.messages) > spec.get('max_trajectory_chars', 1000000):
                    stats['oversize_trajectory'] += 1
                    continue
                split = split_for(t.split_group, seed, validation_fraction)
                splits[split].append(t)
                stats['accepted_' + split] += 1
                audit.write(json.dumps(asdict(t) | {'split': split}, sort_keys=True, ensure_ascii=False) + '\n')
            if not sum(stats['accepted_' + s] for s in splits):
                raise ValueError(f"No accepted trajectories from {spec['path']}: {dict(stats)}")
            sources.append({'spec': spec, 'counts': dict(stats)})
    summary = {}
    for split, records in splits.items():
        episodes, stats = trajectory_episodes(records, tokenizer, cfg, protocol=protocol)
        if not episodes:
            raise ValueError(f'No {split} episodes; increase source sample or use a supported protocol. Counts: {dict(stats)}')
        save_episodes(output / f'{split}.jsonl', episodes)
        summary[split] = dict(stats) | {'trajectories': len(records), 'groups': len({t.split_group for t in records})}
    if {t.split_group for t in splits['train']} & {t.split_group for t in splits['validation']}:
        raise AssertionError('Group leakage')
    manifest = dict(format=1, protocol=protocol, seed=seed, validation_fraction=validation_fraction,
                    sources=sources, budgets=cfg, summary=summary,
                    sha256={name: file_sha256(output / name) for name in ('train.jsonl', 'validation.jsonl', 'normalized.jsonl')},
                    teacher_policy='Recorded assistant targets; no live teacher calls or tool execution.',
                    support_labels='Provided context, not verified sufficient or necessary support.',
                    split_policy='Group-disjoint subsets of the specified upstream split; SWE grouped by repository.',
                    leakage_checks=dict(overlapping_groups=0, exact_trajectory_dedup=True,
                                        near_duplicate_detection=False, external_benchmark_decontamination=False))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
