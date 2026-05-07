import torch
from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import LMDeploywithChatTemplate


with read_base():
    # --- In-domain math ---
    from opencompass.configs.datasets.math.math_prm800k_500_0shot_cot_gen_11c4b5 import math_datasets        # MATH-500
    from opencompass.configs.datasets.aime2024.aime2024_0shot_nocot_gen_2b9dc2 import aime2024_datasets      # AIME 2024
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets                  # GSM8K
    from opencompass.configs.datasets.MathBench.mathbench_2024_gen_50a320 import mathbench_datasets          # MathBench
    # --- General capability ---
    from opencompass.configs.datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets                              # MMLU
    from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import ifeval_datasets                        # IFEval
    from opencompass.configs.datasets.hellaswag.hellaswag_10shot_gen_e42710 import hellaswag_datasets        # HellaSwag
    from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets                       # HumanEval
    from opencompass.configs.datasets.mbpp.sanitized_mbpp_mdblock_0shot_nocot_gen_a2e416 import sanitized_mbpp_datasets  # MBPP
    # --- OOD transferability ---
    from opencompass.configs.datasets.livecodebench.livecodebench_gen_6966bc import LCB_datasets             # LiveCodeBench
    from opencompass.configs.datasets.gpqa.gpqa_gen_4baadb import gpqa_datasets                              # GPQA Diamond
    # --- Summarizer ---
    from opencompass.configs.summarizers.groups.mmlu import mmlu_summary_groups
    from opencompass.configs.summarizers.groups.mathbench_v1_2024 import mathbench_2024_summary_groups

# ── Summarizer ──────────────────────────────────────────────────────────────
summary_groups = sum(
    [v for k, v in locals().items() if k.endswith('_summary_groups')], []
)

summarizer = dict(
    dataset_abbrs=[
        '=== In-domain Math ===',
        ['math_prm800k_500',  'accuracy'],
        ['aime2024',          'accuracy'],
        ['gsm8k',             'accuracy'],
        ['Mathbench',         'naive_average'],
        '',
        '=== General Capability ===',
        ['mmlu',                            'naive_average'],
        ['IFEval',                          'Prompt-level-strict-accuracy'],
        ['hellaswag',                        'accuracy'],
        ['openai_humaneval',                'humaneval_pass@1'],
        ['sanitized_mbpp',                  'score'],
        '',
        '=== OOD Transferability ===',
        ['livecodebench_code_generation',   'pass@1'],
        ['GPQA_diamond',                    'accuracy'],
        '',
        '--- MMLU breakdown ---',
        'mmlu', 'mmlu-stem', 'mmlu-social-science', 'mmlu-humanities', 'mmlu-other',
        '',
        '--- MathBench breakdown ---',
        'mathbench-a (average)', 'mathbench-t (average)',
    ],
    summary_groups=summary_groups,
)

# ── Datasets ─────────────────────────────────────────────────────────────────
datasets = [
    # in-domain math
    *math_datasets,
    *aime2024_datasets,
    *gsm8k_datasets,
    *mathbench_datasets,
    # general capability
    *mmlu_datasets,
    *ifeval_datasets,
    *hellaswag_datasets,
    *humaneval_datasets,
    *sanitized_mbpp_datasets,
    # OOD transferability
    *LCB_datasets,
    *gpqa_datasets,
]

for dataset in datasets:
    dataset['infer_cfg']['inferencer']['batch_size'] = 128

# ── Models: SFT checkpoints ──────────────────────────────────────────────────
CKPT_ROOT = '/home/ubuntu/uq/checkpoints'

# (abbr, path, block_length, confidence_threshold)
model_configs = [
    ('sdar-sft-step020', f'{CKPT_ROOT}/sdar-4b-step20-base',                              4, 1.0),
    ('sdar-sft-step040', f'{CKPT_ROOT}/checkpoint-40',    4, 1.0),
    ('sdar-sft-step060', f'{CKPT_ROOT}/checkpoint-60',  4, 1.0),
    ('sdar-sft-step080', f'{CKPT_ROOT}/checkpoint-80',  4, 1.0),
    ('sdar-sft-step100', f'{CKPT_ROOT}/checkpoint-100', 4, 1.0),
    ('sdar-sft-step120', f'{CKPT_ROOT}/checkpoint-120', 4, 1.0),
]

models = []
for abbr, path, block_length, threshold in model_configs:
    unmasking_strategy = 'low_confidence_static' if threshold == 1.0 else 'low_confidence_dynamic'
    models.append(
        dict(
            type=LMDeploywithChatTemplate,
            abbr=abbr,
            path=path,
            run_cfg=dict(num_gpus=1),
            generation_kwargs=dict(
                top_p=0.95,
                top_k=50,
                temperature=1.0,
                do_sample=False,
                max_new_tokens=4096,
            ),
            model_kwargs=dict(
                tp=1,
                dtype='float16',
                dllm_block_length=block_length,
                dllm_denoising_steps=block_length,
                dllm_confidence_threshold=threshold,
                dllm_unmasking_strategy=unmasking_strategy,
            ),
        )
    )

# ── Runner ────────────────────────────────────────────────────────────────────
# 1 A40 → run one (model, dataset) shard at a time.
# Increase max_num_workers if you get access to more GPUs.
GPUS = 1

infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=GPUS,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=GPUS,
        keep_tmp_file=True,
        task=dict(type=OpenICLInferTask),
        retry=5,
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=1),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)

work_dir = '/home/ubuntu/uq/opencompass/outputs/eval-sdar-sft-math'
