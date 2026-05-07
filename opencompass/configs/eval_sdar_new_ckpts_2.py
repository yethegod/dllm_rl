from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import LMDeploywithChatTemplate


with read_base():
    # --- In-domain math ---
    from opencompass.configs.datasets.math.math_prm800k_500_0shot_cot_gen_11c4b5 import math_datasets
    from opencompass.configs.datasets.aime2024.aime2024_0shot_nocot_gen_2b9dc2 import aime2024_datasets
    # --- General capability ---
    from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import ifeval_datasets
    from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets
    # --- OOD transferability ---
    from opencompass.configs.datasets.livecodebench.livecodebench_gen_6966bc import LCB_datasets
    from opencompass.configs.datasets.gpqa.gpqa_gen_4baadb import gpqa_datasets

summarizer = dict(
    dataset_abbrs=[
        '=== In-domain Math ===',
        ['math_prm800k_500',  'accuracy'],
        ['aime2024',          'accuracy'],
        '',
        '=== General Capability ===',
        ['IFEval',                          'Prompt-level-strict-accuracy'],
        ['openai_humaneval',                'humaneval_pass@1'],
        '',
        '=== OOD Transferability ===',
        ['lcb_code_generation',             'pass@1'],
        ['GPQA_diamond',                    'accuracy'],
    ],
)

# ── Datasets ─────────────────────────────────────────────────────────────────
datasets = [
    *math_datasets,
    *aime2024_datasets,
    *ifeval_datasets,
    *humaneval_datasets,
    *LCB_datasets,
    *gpqa_datasets,
]

for dataset in datasets:
    dataset['infer_cfg']['inferencer']['batch_size'] = 512

# ── Models: new_checkpoints_2 ─────────────────────────────────────────────────
NEW_CKPT_ROOT = '/lambda/nfs/uq/new_checkpoints_2'

model_configs = [
    ('sdar-new2-step010', f'{NEW_CKPT_ROOT}/checkpoint-10', 4, 1.0),
    ('sdar-new2-step020', f'{NEW_CKPT_ROOT}/checkpoint-20', 4, 1.0),
    ('sdar-new2-step030', f'{NEW_CKPT_ROOT}/checkpoint-30', 4, 1.0),
    ('sdar-new2-step040', f'{NEW_CKPT_ROOT}/checkpoint-40', 4, 1.0),
    ('sdar-new2-step050', f'{NEW_CKPT_ROOT}/checkpoint-50', 4, 1.0),
    ('sdar-new2-step060', f'{NEW_CKPT_ROOT}/checkpoint-60', 4, 1.0),
    ('sdar-new2-step070', f'{NEW_CKPT_ROOT}/checkpoint-70', 4, 1.0),
    ('sdar-new2-step080', f'{NEW_CKPT_ROOT}/checkpoint-80', 4, 1.0),
    ('sdar-new2-step090', f'{NEW_CKPT_ROOT}/checkpoint-90', 4, 1.0),
    ('sdar-new2-step100', f'{NEW_CKPT_ROOT}/checkpoint-100', 4, 1.0),
    ('sdar-new2-step110', f'{NEW_CKPT_ROOT}/checkpoint-110', 4, 1.0),
    ('sdar-new2-step120', f'{NEW_CKPT_ROOT}/checkpoint-120', 4, 1.0),
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
                max_batch_size=512,
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

work_dir = '/home/ubuntu/uq/opencompass/outputs/eval-sdar-new-ckpts-2'
