from apertools.parsers import Sentinel
from pathlib import Path


def get_sentinel_times(sentinel_path=".", output="./goes_data/"):
    results = Path(sentinel_path).glob("*.SAFE")
    all_sents = sorted([Sentinel(r) for r in results])
    # using only 1 Sentinel per date
    sents = []
    dates = set()
    for s in all_sents:
        if s.date not in dates:
            dates.add(s.date)
            sents.append(s)
    return sents