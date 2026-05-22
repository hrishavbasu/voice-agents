from services.llm_splitting import limit_first_clause


def test_limit_first_clause_splits_long_first_sentence():
    long = " ".join(["word"] * 20) + "?"
    parts = limit_first_clause(long, is_first=True)
    assert len(parts[0].split()) <= 15
    assert len(parts) >= 2


def test_limit_first_clause_unchanged_when_short():
    short = "Hello, how can I help?"
    assert limit_first_clause(short, is_first=True) == [short]
