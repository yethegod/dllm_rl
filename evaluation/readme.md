# Evaluation

We use [SDAR](https://github.com/JetAstra/SDAR) and [TraceRL](https://github.com/Gen-Verse/dLLM-RL) to evaluate the fine-tuned models.

## Environment setup

Use separate Conda environments for SDAR/OpenCompass and dLLM-RL evaluation. The two toolchains pin different CUDA/PyTorch-related packages, so sharing one environment is likely to cause dependency conflicts.

### SDAR / OpenCompass evaluation

The SDAR evaluation code uses the OpenCompass setup under `evaluation/SDAR/evaluation`.

```bash
cd evaluation/SDAR/evaluation

# Create the OpenCompass environment from the provided lock file.
conda env create -f environment.yml
conda activate opencompass

# Run OpenCompass from its local source tree.
cd opencompass
```

Sanity check:

```bash
python -c "import torch, transformers, lmdeploy; print(torch.__version__)"
```

Then run an SDAR evaluation from `evaluation/SDAR/evaluation/opencompass`:

```bash
# lmdeploy backend
python run.py configs/eval_sdar_lmdeploy.py

# Hugging Face backend
python run.py configs/eval_sdar_hf.py
```

Notes:

- The provided `environment.yml` creates an environment named `opencompass`.
- It pins a Linux CUDA stack, including CUDA 12.8 PyTorch wheels and `lmdeploy`. If the CUDA-specific wheels are unavailable on your machine, install the matching PyTorch stack for your CUDA driver first, then reinstall the remaining packages from `environment.yml`.
- If you want to call OpenCompass outside the local `opencompass` directory, install it in editable mode:

```bash
pip install -e .
```

### dLLM-RL / TraceRL evaluation

The dLLM-RL evaluation code uses the TraceRL environment under `evaluation/dLLM-RL`.

```bash
cd evaluation/dLLM-RL

conda create --name dllm-rl python=3.10 -y
conda activate dllm-rl

pip install torch==2.6.0 torchvision==0.21.0
pip install --no-cache-dir \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -r requirements.txt
```

Sanity check:

```bash
python -c "import torch, transformers, accelerate; print(torch.__version__)"
```

Then run evaluation from `evaluation/dLLM-RL`:

```bash
python eval.py config=configs/sdar_eval.yaml
```

Notes:

- The flash-attn wheel above is for Linux, Python 3.10, CUDA 12, and torch 2.6.0. For a different CUDA/Python/PyTorch setup, install the matching `flash-attn` build instead.

## Usage

This project supports two evaluation paths:

- **SDAR / OpenCompass**: use this path for OpenCompass-style benchmark evaluation with SDAR-specific inference backends.
- **dLLM-RL / TraceRL**: use this path for TraceRL-style diffusion language model evaluation, especially when evaluating the fine-tuned SDAR checkpoint from this project.

### Configuration

#### SDAR / OpenCompass

Edit one of the following config files:

- `evaluation/SDAR/evaluation/opencompass/configs/eval_sdar_lmdeploy.py`
- `evaluation/SDAR/evaluation/opencompass/configs/eval_sdar_hf.py`

Update these fields:

1. **Set up your models**: edit the `model_configs` list. Each entry has the format `(name, model_path, block_length, threshold, num_gpus)`.
2. **Set unique names**: use the first item in each `model_configs` tuple as the model nickname shown in OpenCompass results.
3. **Configure GPU settings**: set `num_gpus` per model and update the global `GPUS` value to match the number of GPUs available on the machine.
4. **Configure generation length**:
   - lmdeploy config: edit `generation_kwargs.max_new_tokens`.
   - Hugging Face config: edit `generation_kwargs.gen_length`.
5. **Select benchmarks**: edit the `datasets` list. The provided configs include common benchmarks such as GSM8K, MATH, HumanEval, MBPP, MMLU, MathBench, and IFEval.
6. **Set output directory**: edit `work_dir` if you want results written somewhere other than `./outputs/eval-chat-sdar`.

#### dLLM-RL / TraceRL

Edit:

- `evaluation/dLLM-RL/configs/sdar_eval.yaml`

Update these fields:

1. **Set up your model**: set `model` to your fine-tuned SDAR checkpoint path or Hugging Face repository name.
2. **Set model type**: keep `model_base: "sdar"` for SDAR and TraDo-style block diffusion models.
3. **Select benchmark**: set `dataset.eval_dataset` to one of the supported datasets, such as `MATH500`, `GSM8K`, `AIME2024`, `GPQA`, `HumanEval`, `MBPP`, `LiveCodeBench`, or `LiveBench`.
4. **Set dataset type**: set `dataset.data_type` to `math` for math/general QA benchmarks or `code` for code-generation benchmarks.
5. **Configure GPU settings**: adjust `rollout.tensor_parallel_size` and `rollout.max_active` based on GPU memory. If evaluation runs out of memory, reduce `max_active` first; if needed, increase `tensor_parallel_size`.
6. **Configure generation length**: set `rollout.max_token` to the maximum number of generated tokens.
7. **Configure decoding**: adjust `rollout.block_size`, `rollout.denoising_steps_per_block`, `temperature`, `top_p`, `top_k`, and `remasking_strategy` as needed.

### Model Path Format

Model paths can be either:

- A local checkpoint path:

```bash
/path/to/your/local/model
```

- A Hugging Face repository name:

```bash
yethegod/sdar-4b-rl
```

For OpenCompass, place the path in `model_configs`. For dLLM-RL / TraceRL, place it in the `model` field of `configs/sdar_eval.yaml`.


### Running Evaluations

#### SDAR / OpenCompass

Run from `evaluation/SDAR/evaluation/opencompass`:

```bash
cd evaluation/SDAR/evaluation/opencompass

# lmdeploy backend
python run.py configs/eval_sdar_lmdeploy.py

# Hugging Face backend
python run.py configs/eval_sdar_hf.py
```

#### dLLM-RL / TraceRL

Run from `evaluation/dLLM-RL`:

```bash
cd evaluation/dLLM-RL

python eval.py config=configs/sdar_eval.yaml
```

For multi-node evaluation, use the corresponding multi-node config:

```bash
python multinode_eval.py config=configs/sdar_multinode_eval.yaml
```

### Generation Length Guidelines

- **Short or non-reasoning benchmarks**: start with `max_new_tokens`, `gen_length`, or `rollout.max_token` around `1024` to `2048`.
- **Math reasoning benchmarks**: use a larger limit such as `4096` or higher if the model produces long reasoning traces.
- **Long reasoning or code benchmarks**: increase the limit as needed, but monitor GPU memory and runtime.
- **If out of memory occurs**: reduce batch size or `rollout.max_active` before reducing generation length.
