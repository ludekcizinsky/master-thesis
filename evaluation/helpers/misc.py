from typing import Iterable, List

def chunked(seq: List[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]