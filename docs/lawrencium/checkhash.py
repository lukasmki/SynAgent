"""Recompute axolotl's dataset cache hash for timing.yaml and compare it to
what the preprocess pass actually wrote on disk.

Cannot just call load_cfg(): it validates bf16 against a visible GPU and this
runs on the login node. So replicate normalize_config's two relevant defaults
(eval_sample_packing mirrors sample_packing; tokenizer_config falls back to
base_model) and the md5 formula from utils/data/sft.py:149.
"""
import sys, yaml, hashlib
from pathlib import Path

cfg = yaml.safe_load(open(sys.argv[1]))

sample_packing = cfg.get("sample_packing")
eval_sample_packing = cfg.get("eval_sample_packing")
if eval_sample_packing is None:          # check_eval_packing() mirrors it
    eval_sample_packing = sample_packing
tokenizer_name = cfg.get("tokenizer_config") or cfg["base_model"]

parts = []
for d in cfg["datasets"]:
    path = d.get("path")
    typ = d.get("type")
    shards = d.get("shards")
    conversation = d.get("conversation")
    split = d.get("split", "train")      # DatasetConfig default
    parts.append(f"{path}:{typ}:{shards}:{conversation}{split}")

s = (
    f"{cfg.get('sequence_len')}@{sample_packing}@{eval_sample_packing}"
    f"@{cfg.get('group_by_length')}@" + "|".join(sorted(parts)) + "|" + tokenizer_name
)
h = hashlib.md5(s.encode(), usedforsecurity=False).hexdigest()

print(f"config       : {sys.argv[1]}")
print(f"hash string  : {s}")
print(f"computed hash: {h}")

prep = Path(cfg["dataset_prepared_path"])
on_disk = sorted(p.name for p in prep.iterdir()) if prep.is_dir() else []
print(f"on disk      : {on_disk}")
print()
print("MATCH -- training run will hit the cache" if h in on_disk
      else "MISMATCH -- training run would re-tokenize")
