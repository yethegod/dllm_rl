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

