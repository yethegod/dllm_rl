"""Quick test: compare confidence_threshold 1.0 / 0.9 / 0.8 on GSM8K subset."""
import json, re, multiprocessing, os
multiprocessing.set_start_method('fork', force=True)
os.environ['TORCHDYNAMO_DISABLE'] = '1'  # disable CUDA graph capture

from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig

CKPT = '/home/ubuntu/uq/checkpoints/checkpoint-80'
N    = 50  # samples to test

# Load GSM8K test split from local cache
OFFSET = 200  # start from sample 200
with open('/home/ubuntu/uq/opencompass_data/data/gsm8k/test.jsonl') as f:
    all_samples = [json.loads(l)['question'] for l in f]
samples = all_samples[OFFSET:OFFSET + N]
prompts = [
    f"{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    for q in samples
]

def is_garbage(text):
    t = text.strip()
    if not t: return True
    return not re.sub(r'(okino\s*)', '', t).strip()

gen_cfg = GenerationConfig(
    max_new_tokens=512,
    do_sample=False,
    stop_words=['<|im_end|>', '<|endoftext|>'],
)

if __name__ == '__main__':
    print(f"{'Threshold':<12} {'Empty':>6} {'Garbage':>8} {'Valid':>6} {'Valid%':>7}")
    print('-' * 45)

    for threshold in [1.0, 0.8, 0.5]:
        engine_cfg = PytorchEngineConfig(
            tp=1,
            dtype='float16',
            eager_mode=True,
            dllm_block_length=4,
            dllm_denoising_steps=4,
            dllm_confidence_threshold=threshold,
            dllm_unmasking_strategy='low_confidence_static',
        )
        pipe = pipeline(CKPT, backend_config=engine_cfg)
        outputs = pipe(prompts, gen_config=gen_cfg)
        texts = [o.text for o in outputs]
        del pipe

        empty   = sum(1 for t in texts if not t.strip())
        garbage = sum(1 for t in texts if is_garbage(t))
        valid   = N - garbage
        print(f"{threshold:<12} {empty:>6} {garbage:>8} {valid:>6} {valid/N*100:>6.0f}%")
        # Show a sample
        sample = next((t for t in texts if not is_garbage(t)), None)
        if sample:
            print(f"  sample: {repr(sample[:120])}")
