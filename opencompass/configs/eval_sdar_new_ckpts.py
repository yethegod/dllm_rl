import torch
from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import LMDeploywithChatTemplate


with read_base():
    # --- In-domain math ---
    from opencompass.configs.datasets.math.math_prm800k_500_0shot_cot_gen_11c4b5 import math_datasets
    from opencompass.configs.datasets.aime2024.aime2024_0shot_nocot_gen_2b9dc2 import aime2024_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets
    from opencompass.configs.datasets.MathBench.mathbench_2024_gen_50a320 import mathbench_datasets
    # --- General capability ---
    from opencompass.configs.datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets
    from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import ifeval_datasets
    from opencompass.configs.datasets.hellaswag.hellaswag_10shot_gen_e42710 import hellaswag_datasets
    from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets
    from opencompass.configs.datasets.mbpp.sanitized_mbpp_mdblock_0shot_nocot_gen_a2e416 import sanitized_mbpp_datasets
    # --- OOD transferability ---
    from opencompass.configs.datasets.livecodebench.livecodebench_gen_6966bc import LCB_datasets
    from opencompass.configs.datasets.gpqa.gpqa_gen_4baadb import gpqa_datasets
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
    *math_datasets,
    *aime2024_datasets,
    *gsm8k_datasets,
    *mathbench_datasets,
    *mmlu_datasets,
    *ifeval_datasets,
    *hellaswag_datasets,
    *humaneval_datasets,
    *sanitized_mbpp_datasets,
    *LCB_datasets,
    *gpqa_datasets,
]

for dataset in datasets:
    dataset['infer_cfg']['inferencer']['batch_size'] = 128

# ── Models: new checkpoints ───────────────────────────────────────────────────
NEW_CKPT_ROOT = '/home/ubuntu/uq/new_checkpoints'

model_configs = [
    ('sdar-new-step010', f'{NEW_CKPT_ROOT}/checkpoint-10', 4, 1.0),
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

work_dir = '/home/ubuntu/uq/opencompass/outputs/eval-sdar-new-ckpts'
