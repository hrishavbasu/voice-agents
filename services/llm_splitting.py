"""Text splitting helpers for LLM streaming (no provider imports)."""
from typing import List

_MAX_FIRST_CLAUSE_WORDS = 15


def limit_first_clause(sentence: str, is_first: bool) -> List[str]:
    if not is_first or len(sentence.split()) <= _MAX_FIRST_CLAUSE_WORDS:
        return [sentence]
    words = sentence.split()
    first = " ".join(words[:_MAX_FIRST_CLAUSE_WORDS])
    rest = " ".join(words[_MAX_FIRST_CLAUSE_WORDS:])
    return [first, rest] if rest.strip() else [first]
