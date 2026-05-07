# Exploration in Diffusion Language Model Reasoning Transferability
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co/yethegod/sdar-4b-rl)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/yethegod/dllm_rl) 

This is the codebase for CSCI 5980 course project: Exploration in Diffusion Language Model Reasoning Transferability.

## Overview
While prior work has extensively studied reasoning transferability in autoregressive LLMs, it remains unclear whether diffusion language models exhibit similar transfer patterns. This project investigates that question by evaluating how reasoning-oriented training transfers across mathematical reasoning, broader reasoning tasks, and non-reasoning benchmarks.

### Compute Resources

- **RL training**: 8 x NVIDIA A100 40GB GPUs.
- **Inference / evaluation**: 2 x NVIDIA A100 40GB GPUs.
- **SFT training**: TBD.

### Benchmark Categories

| Category | Benchmarks | Description |
|----------|------------|-------------|
| **Math Reasoning** | MATH-500, AIME24 | Evaluates mathematical problem-solving ability, including multi-step reasoning, symbolic manipulation, and competition-style quantitative questions. |
| **Other Reasoning** | GPQA-Diamond, HumanEval | Evaluates reasoning beyond mathematics, including graduate-level scientific QA and code-generation tasks that require planning and logical consistency. |
| **Non-Reasoning** | Hellaswag, IFEval | Evaluates general language understanding and instruction-following ability, serving as a check that reasoning transfer does not degrade broader model behavior. |

### Findings
| Method | Math Reasoning | Other Reasoning | Non-Reasoning |
|--------|---------------|----------------|---------------|
| **RL** | ✅ Strong gains | ✅ Positive transfer | ✅ Preserved/improved |

## Acknowledgements and Licenses

This project builds on two open-source codebases:

- [SDAR](https://github.com/JetAstra/SDAR), which is released under the [MIT License](evaluation/SDAR/LICENSE). We use SDAR-4B-Chat as the base diffusion language model and for SDAR-specific evaluation components.
- [dLLM-RL / TraceRL](https://github.com/Gen-Verse/dLLM-RL), which is released under the [Apache License 2.0](evaluation/dLLM-RL/LICENSE). We use its reinforcement learning and evaluation infrastructure for diffusion language models.

We thank the authors and contributors of SDAR and dLLM-RL / TraceRL for releasing their code and models to the community. Their work provides the foundation for this course project.
