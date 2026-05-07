# SDAR new_checkpoints_2 — Consolidated Evaluation Summary

**Model**: SDAR (Qwen2-4B base), discrete diffusion LM  
**Inference**: `block_length=4`, `denoising_steps=4`, `threshold=1.0`, `max_new_tokens=4096`  
**Checkpoints**: `/lambda/nfs/uq/new_checkpoints_2`

## Results

| Metric | step10 | step20 | step30 | step40 | step50 | step60 | step70 | step80 | step90 | step100 | step110 | step120 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **MATH 500** | 61.00 | 70.00 | 67.60 | 69.20 | 72.40 | 70.60 | 71.80 | 74.20 | 73.20 | 72.00 | 72.60 | **74.00** |
| **AIME 2024** | 10.00 | 6.67 | 6.67 | 0.00 | 10.00 | 3.33 | 6.67 | 6.67 | 10.00 | **13.33** | 6.67 | 10.00 |
| **IFEval** | **57.49** | 54.16 | 53.42 | 52.31 | 50.09 | 52.50 | 49.17 | 51.02 | 50.28 | 51.76 | 52.50 | 48.80 |
| **HumanEval** | 74.39 | 78.66 | **80.49** | 79.88 | 78.66 | 78.05 | 75.61 | 78.05 | 77.44 | 79.88 | 76.83 | 78.05 |
| **LiveCodeBench** | **25.75** | 22.25 | 18.00 | 16.50 | 17.00 | 17.75 | 19.25 | 18.75 | 20.25 | 23.00 | 20.50 | 20.75 |
| **GPQA Diamond** | 33.33 | 33.33 | 36.36 | 38.38 | 37.88 | 36.87 | 37.88 | 36.87 | 39.39 | **40.40** | 38.89 | 36.87 |

## Key Observations

- **MATH500**: Best at step80 (74.2%) and step120 (74.0%) — improves with training
- **AIME**: Best at step100 (13.3%) — noisy metric (only 30 samples)
- **IFEval**: Steadily declines — instruction following hurt by math SFT
- **HumanEval**: Peaks at step30 (80.5%), then degrades slightly
- **LiveCodeBench**: Drops sharply early (step40: 16.5%), partially recovers later
- **GPQA**: Best at step100 (40.4%), general upward trend through training
