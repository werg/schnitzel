from pathlib import Path
import runpy

from schnitz.data import Source


class WordTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split()) + int(add_special_tokens)))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize and add_generation_prompt
        return messages[0]['content'] + '\nAssistant:'


def helpers():
    return runpy.run_path(str(Path(__file__).parents[1] /
                              'scripts/prepare_hotpot_curriculum.py'))


def test_hotpot_chunks_preserve_sentence_support_mapping_and_budget():
    chunk = helpers()['paragraph_chunks']
    tokenizer = WordTokenizer()
    sources, mapping = chunk('Title', [
        'one two three four five six seven eight',
        'nine ten eleven twelve',
    ], tokenizer, 12)
    assert len(sources) == 2
    assert mapping == {0: [sources[0].record_id], 1: [sources[1].record_id]}
    assert all(len(tokenizer.encode(source.text, add_special_tokens=True)) <= 12
               for source in sources)


def test_hotpot_chunks_split_a_single_oversized_token():
    chunk = helpers()['paragraph_chunks']
    tokenizer = WordTokenizer()
    # The real tokenizer may split one whitespace token into many subwords.
    tokenizer.encode = lambda text, add_special_tokens=False: list(range(
        len(text) // 4 + int(add_special_tokens)))
    sources, mapping = chunk('Title', ['x' * 100], tokenizer, 12)
    assert len(sources) > 1
    assert mapping[0] == [source.record_id for source in sources]
    assert all(len(tokenizer.encode(source.text, add_special_tokens=True)) <= 12
               for source in sources)


def test_hotpot_episode_keeps_only_prior_verified_supports_and_hides_answer():
    module = helpers()
    tokenizer = WordTokenizer()
    first = module['paragraph_chunks']('First', ['alpha answer fact.'], tokenizer, 20)
    second = module['paragraph_chunks']('Second', ['beta bridge fact.'], tokenizer, 20)
    decoy = module['paragraph_chunks']('Decoy', ['irrelevant material.'], tokenizer, 20)
    row = {'id': 'q1', 'question': 'What is linked?', 'answer': 'answer',
           'type': 'bridge', 'level': 'hard',
           'supporting_facts': {'title': ['First', 'Second'], 'sent_id': [0, 0]}}
    episode = module['_episode'](row, {'First': first, 'Second': second, 'Decoy': decoy},
        split='train', max_supports=4, max_prompt_tokens=64, max_target_tokens=16,
        tokenizer=tokenizer)
    assert episode is not None
    assert len(episode.required_ids) == 2
    assert episode.sufficient_groups == (episode.required_ids,)
    assert len(episode.supports) == 3
    assert all(source.created_at < episode.query_time for source in episode.supports)
    assert episode.answer not in episode.query


def test_reconstruction_mix_excludes_validation_source_labels():
    mix = helpers()['_reconstruction_mix']
    sources = {name: Source(
        name, f'Title: {name}\nPassage: ' + ' '.join(f'w{i}' for i in range(20)),
        1, 'passage') for name in ('train-a', 'heldout', 'train-b')}
    episodes = mix(sources, {name: name for name in sources}, {'heldout'}, 3)
    required = {rid for episode in episodes for rid in episode.required_ids}
    assert required == {'train-a', 'train-b'}
