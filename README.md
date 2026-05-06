# Exploration in Diffusion Language Model Reasoning Transferability
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co/yethegod/sdar-4b-rl)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/yethegod/dllm_rl) 

This is the codebase for CSCI 5980 course project: Exploration in Diffusion Language Model Reasoning Transferability.

## Overview
While prior work has extensively studied reasoning transferability in autoregressive LLMs, it remains unclear whether diffusion language models exhibit similar transfer patterns. This project investigates that question by evaluating how reasoning-oriented training transfers across mathematical reasoning, broader reasoning tasks, and non-reasoning benchmarks.

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