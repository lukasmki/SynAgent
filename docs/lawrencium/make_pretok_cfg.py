#!/usr/bin/env python
"""Derive a CPU-safe preprocessing config from timing.yaml.

WHY DERIVE INSTEAD OF HAND-COPY
axolotl caches the tokenized dataset under a directory named for an md5 of:

    sequence_len @ sample_packing @ eval_sample_packing @ group_by_length
    @ <dataset path:type:shards:conversation+split ...> | tokenizer_config

If any of those drift between the preprocess pass and the training pass, the
GPU job gets a cache MISS and re-tokenizes 1M rows anyway -- exactly what this
is meant to prevent. Loading the real config and mutating only non-hash keys
makes that drift impossible.

WHAT IS CHANGED, AND WHY IT IS SAFE
  bf16/tf32/fp16 -> False   Tokenization has no dtype. axolotl 0.6.0 rejects
                            bf16 when no Ampere GPU is visible; it exempts
                            `is_preprocess`, but sets that flag in
                            cli/preprocess.py:35 AFTER load_cfg() validates on
                            line 34, so the exemption never fires. Upstream bug.
  fsdp/fsdp_config -> gone  Sharding is a training concern; on a CPU node the
                            distributed init has nothing to attach to.

None of these five keys appear in the hash, so the cache directory is identical.
"""

import sys
import yaml

src, dst = sys.argv[1], sys.argv[2]

with open(src) as f:
    cfg = yaml.safe_load(f)

# Snapshot the hash-relevant fields so any accidental change is caught loudly
# rather than showing up as a silent re-tokenization hours later.
def key_fields(c):
    return {
        "sequence_len": c.get("sequence_len"),
        "sample_packing": c.get("sample_packing"),
        "eval_sample_packing": c.get("eval_sample_packing"),
        "group_by_length": c.get("group_by_length"),
        "datasets": c.get("datasets"),
        "dataset_prepared_path": c.get("dataset_prepared_path"),
        "base_model": c.get("base_model"),
        "tokenizer_config": c.get("tokenizer_config"),
    }

before = key_fields(cfg)

for k in ("bf16", "tf32", "fp16", "bfloat16", "float16"):
    if k in cfg:
        cfg[k] = False
cfg["fp16"] = False
cfg.pop("fsdp", None)
cfg.pop("fsdp_config", None)

after = key_fields(cfg)
assert before == after, f"hash-relevant field changed!\n{before}\n{after}"

with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("wrote", dst)
print("cache-key fields (must match the training run):")
for k, v in after.items():
    print(f"  {k}: {v}")
