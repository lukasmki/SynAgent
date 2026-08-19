#!/usr/bin/env python
"""Build a training subset from lukaskim/cadd-instruct-raw.

WHY NOT config="all"
--------------------
`all/` on the Hub holds two overlapping exports and the config globs
`all/*.parquet`, so it loads both:

    all/train-00000-of-00001.parquet   3,856,395 rows  <- STALE
    all/train-0000{0..8}-of-00009.parquet  5,856,395 rows  <- current

Verified from the parquet footers and the `source` column: the stale file is
smileyllama (2,330,226) + linkllama (1,526,169) with NO synllama rows -- it
predates synllama being added. Loading `all` therefore counts smileyllama and
linkllama twice and synllama once, shifting the mix from 40/26/34 to 48/31/21
and under-weighting synthesis planning by a third.

So we read the three per-source directories explicitly. Real total: 5,856,395.

WHY PYARROW AND NOT datasets.load_dataset
-----------------------------------------
The first version used `load_dataset(...)` then
`ds.select(rng.sample(range(len(ds)), n))`. Random-access selection across
millions of rows in the datasets library is pathologically slow -- it ran for
over 35 minutes on 1M rows without producing output, while holding an idle A40.

Reading the parquet shards directly with pyarrow and slicing after a shuffled
take is I/O-bound rather than random-access-bound, and completes in minutes.

Usage
-----
    python 01_prepare_data.py --rows 1000000 --out $SCRATCH/cadd/train_1M.jsonl
    python 01_prepare_data.py --rows 2000    --out $SCRATCH/cadd/smoke.jsonl
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

REPO = "lukaskim/cadd-instruct-raw"
SOURCES = ["smileyllama", "linkllama", "synllama"]
COLS = ["instruction", "input", "output", "source"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1_000_000,
                    help="Total rows to emit. 0 means everything.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--balanced", action="store_true",
                    help="Equal rows per source instead of natural proportions.")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Fetch only the three per-source directories. Deliberately NOT "all/*",
    # which would drag in the stale shard described above.
    print("downloading parquet shards (skipping the stale all/ export)...", flush=True)
    local = snapshot_download(
        REPO,
        repo_type="dataset",
        allow_patterns=[f"{s}/*.parquet" for s in SOURCES],
    )
    root = Path(local)

    tables = {}
    for src in SOURCES:
        files = sorted((root / src).glob("*.parquet"))
        tables[src] = pq.ParquetDataset([str(f) for f in files]).read(columns=COLS)
        print(f"{src:12s} {tables[src].num_rows:>9,} rows", flush=True)

    total = sum(t.num_rows for t in tables.values())
    print(f"{'TOTAL':12s} {total:>9,} rows available", flush=True)

    target = args.rows or total
    if target > total:
        raise SystemExit(f"--rows {target:,} exceeds the {total:,} available")

    if args.balanced:
        per = target // len(SOURCES)
        quota = {s: min(per, tables[s].num_rows) for s in SOURCES}
    else:
        quota = {s: round(target * tables[s].num_rows / total) for s in SOURCES}
    drift = target - sum(quota.values())
    if drift:
        quota[max(quota, key=lambda s: quota[s])] += drift

    rng = random.Random(args.seed)
    rows: list[dict] = []
    for src in SOURCES:
        tbl, n = tables[src], quota[src]
        # Take a random contiguous window rather than n scattered indices --
        # same effect for an already-unordered corpus, vastly cheaper.
        if n >= tbl.num_rows:
            chunk = tbl
        else:
            start = rng.randint(0, tbl.num_rows - n)
            chunk = tbl.slice(start, n)
        rows.extend(chunk.to_pylist())
        print(f"took {n:>9,} from {src}", flush=True)

    rng.shuffle(rows)

    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(rows):,} rows -> {args.out} "
          f"({args.out.stat().st_size / 1e6:,.0f} MB)")
    print("mix:", dict(Counter(r["source"] for r in rows)))


if __name__ == "__main__":
    main()
