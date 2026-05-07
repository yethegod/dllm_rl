#!/bin/bash
# Pre-download all eval datasets to scratch.
# Can be run on either ahl01 or agc08 (both have internet access).

export PATH=/users/8/pan00389/miniconda3/envs/opencompass/bin:$PATH
export HF_HOME=/scratch.global/pan00389/hf_cache
export HF_DATASETS_CACHE=/scratch.global/pan00389/hf_cache/datasets
export HUGGINGFACE_HUB_CACHE=/scratch.global/pan00389/hf_cache/hub
export COMPASS_DATA_CACHE=/scratch.global/pan00389/opencompass_data

mkdir -p /scratch.global/pan00389/hf_cache/datasets
mkdir -p /scratch.global/pan00389/hf_cache/hub
mkdir -p /scratch.global/pan00389/opencompass_data

EVAL_DIR=/users/8/pan00389/dllm/SDAR/evaluation/opencompass
cd $EVAL_DIR

echo "=== Downloading HuggingFace datasets ==="
/users/8/pan00389/miniconda3/envs/opencompass/bin/python - <<'EOF'
from datasets import load_dataset

to_download = [
    ("jnanliu/orz-math-filtered",   dict(split="train")),
    ("openai/openai_humaneval",      dict(split="test")),
    ("Rowan/hellaswag",              dict(split="validation")),
    ("cais/mmlu",                    dict(name="all", split="test")),
    # gsm8k and mbpp are OSS datasets, handled below
]

for path, kwargs in to_download:
    print(f"  {path} ...", flush=True)
    try:
        ds = load_dataset(path, **kwargs)
        print(f"    OK — {len(ds)} examples")
    except Exception as e:
        print(f"    FAILED: {e}")
EOF

echo ""
echo "=== Downloading OpenCompass OSS datasets ==="
/users/8/pan00389/miniconda3/envs/opencompass/bin/python - <<'EOF'
import sys, os
sys.path.insert(0, '/users/8/pan00389/dllm/SDAR/evaluation/opencompass')
from opencompass.utils import get_data_path

# GPQA and IFEval are bundled in ./data/ — no download needed
# GSM8K and MBPP come from HF (opencompass org) — handled above
oc_datasets = [
    ('opencompass/math',                False),
    ('opencompass/aime2024',            False),
    ('opencompass/code_generation_lite',False),
    ('opencompass/gsm8k',               False),
    ('opencompass/sanitized_mbpp',      False),
    ('data/mathbench_v1/test',          True),   # local_mode=True to match MathBenchDataset.load()
]

for ds_id, local_mode in oc_datasets:
    print(f"  {ds_id} ...", flush=True)
    try:
        path = get_data_path(ds_id, local_mode=local_mode)
        print(f"    OK -> {path}")
    except Exception as e:
        print(f"    FAILED: {e}")
EOF

echo ""
echo "=== All downloads complete ==="
echo "HF cache:         /scratch.global/pan00389/hf_cache/datasets/"
echo "OpenCompass data: /scratch.global/pan00389/opencompass_data/data/"
