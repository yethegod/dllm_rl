from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import LMDeploywithChatTemplate

with read_base():
    from opencompass.configs.datasets.aime2024.aime2024_0shot_nocot_gen_2b9dc2 import aime2024_datasets

datasets = [*aime2024_datasets]

for dataset in datasets:
    dataset['infer_cfg']['inferencer']['batch_size'] = 512

NEW_CKPT_ROOT = '/lambda/nfs/uq/new_checkpoints_2'

models = [
    dict(
        type=LMDeploywithChatTemplate,
        abbr='sdar-new2-step040',
        path=f'{NEW_CKPT_ROOT}/checkpoint-40',
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
            dllm_block_length=4,
            dllm_denoising_steps=4,
            dllm_confidence_threshold=1.0,
            dllm_unmasking_strategy='low_confidence_static',
        ),
    )
]

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(
        type=LocalRunner,
        max_num_workers=1,
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

summarizer = dict(
    dataset_abbrs=[
        ['aime2024', 'accuracy'],
    ],
)

work_dir = '/home/ubuntu/uq/opencompass/outputs/eval-sdar-step40-aime-rerun'
