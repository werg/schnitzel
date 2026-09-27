from schnitz.document_ingestion import (grouped_ingestion_prefixes,
    holistic_ingestion_messages, holistic_write_count, prompted_write_messages,
    source_ingestion_groups, split_document, streaming_ingestion_messages,
    writer_prefix_ids)


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        rendered = repr(messages)
        return {'input_ids': list(rendered.encode()), 'assistant_masks': [0] * len(rendered)}


def test_streaming_ingestion_writes_after_every_structured_or_manual_part():
    text = 'First paragraph has facts.\n\n' + ' '.join(f'w{i}' for i in range(30))
    parts = split_document(text, Tokenizer(), max_tokens=10, overlap_words=2)
    assert parts[0].policy == 'paragraph'
    assert any(part.policy == 'token-bounded-overlap' for part in parts)
    messages, records = streaming_ingestion_messages(
        'doc-1', text, Tokenizer(), generation='g1', scope={'domain': 'research'},
        max_part_tokens=10, overlap_words=2)
    assert len(records) == len(parts)
    assert sum(bool(message.get('tool_calls')) for message in messages) == len(parts)
    assert all(record['span_start'] < record['span_end'] for record in records)


def test_holistic_prompt_ends_at_auditable_write_call():
    messages = prompted_write_messages(
        document_id='doc', text='A named fact: 17.', record_id='record', generation='g1',
        scope={'domain': 'research'}, kind='document_fact')
    assert messages[-1]['tool_calls'][0]['function']['name'] == 'memory.write'
    assert writer_prefix_ids(Tokenizer(), messages)
    assert holistic_write_count(128) == 1
    assert holistic_write_count(4096) > holistic_write_count(128)


def test_holistic_ingestion_presents_one_blob_and_separate_model_chosen_writes():
    messages, records = holistic_ingestion_messages(
        'whole-doc', 'Alpha is 17. Beta requires alpha.', generation='g2',
        scope={'domain': 'research'},
        write_contents=('Alpha is exactly 17.', 'Beta requires alpha.'))
    assert sum(message['role'] == 'user' for message in messages) == 1
    calls = [message for message in messages if message.get('tool_calls')]
    assert len(calls) == len(records) == 2
    assert calls[0]['tool_calls'][0]['function']['arguments']['content'] != (
        calls[1]['tool_calls'][0]['function']['arguments']['content'])
    assert all(record['chunking_policy'] == 'holistic-model-decomposition'
               for record in records)


def test_grouped_sources_support_true_holistic_and_streaming_multiwrite_prefixes():
    parts = (('r1', 'First fact.'), ('r2', 'Second fact.'))
    for mode in ('holistic', 'streaming'):
        prefixes = grouped_ingestion_prefixes(
            'doc', parts, generation='g1', scope={'domain': 'research'}, mode=mode)
        assert set(prefixes) == {'r1', 'r2'}
        assert prefixes['r1'][-1]['tool_calls'][0]['function']['arguments']['content'] == 'First fact.'
        assert prefixes['r2'][-1]['tool_calls'][0]['function']['arguments']['content'] == 'Second fact.'
        assert len(prefixes['r2']) > len(prefixes['r1'])
    rows = [
        {'record_id': f'r{i}', 'text': f'part {i}',
         'provenance': {'article_title': 'same'}} for i in range(5)]
    groups = source_ingestion_groups(rows, maximum_parts=4)
    assert groups['r0'][1] == tuple((f'r{i}', f'part {i}') for i in range(4))
    assert groups['r4'][1] == (('r4', 'part 4'),)


def test_source_groups_never_mix_creation_times():
    rows = [{'record_id': f'r{i}', 'text': f'line {i}', 'created_at': 1 + i,
             'provenance': {'article_title': 'story'}} for i in range(3)]
    rows.append({'record_id': 'r9', 'text': 'same time', 'created_at': 2,
                 'provenance': {'article_title': 'story'}})
    groups = source_ingestion_groups(rows, maximum_parts=4)
    assert groups['r0'][1] == (('r0', 'line 0'),) and groups['r0'][0] == 'story#0'
    assert groups['r1'][1] == (('r1', 'line 1'), ('r9', 'same time'))
    assert groups['r1'][0] == 'story@2#0' and groups['r2'][0] == 'story@3#0'
