from schnitz.span_tokens import MEMORY_TOOLS, SENTINEL, SPAN_TOKENS, check_tokenizer, token_id


def test_span_tokens_are_distinct_and_leave_the_sentinel_alone():
    texts = [t for t, _ in SPAN_TOKENS.values()]
    ids = [i for _, i in SPAN_TOKENS.values()]
    assert len(set(texts)) == len(texts) and len(set(ids)) == len(ids)
    assert SENTINEL[0] not in texts and SENTINEL[1] not in ids
    assert token_id('mem_end') == 34
    assert {tool['name'] for tool in MEMORY_TOOLS} == {'memory_search', 'memory_write'}


def test_check_tokenizer_rejects_a_mismatch():
    class Tok:
        def __init__(self, vocab):
            self.vocab = vocab

        def get_vocab(self):
            return self.vocab

    good = {t: i for t, i in SPAN_TOKENS.values()} | {SENTINEL[0]: SENTINEL[1]}
    check_tokenizer(Tok(good))
    bad = dict(good)
    bad[SPAN_TOKENS['mem'][0]] = 99
    try:
        check_tokenizer(Tok(bad))
    except ValueError:
        return
    raise AssertionError('mismatch not detected')
