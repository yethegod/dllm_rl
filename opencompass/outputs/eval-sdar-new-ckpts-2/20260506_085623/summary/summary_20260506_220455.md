| dataset | version | metric | mode | sdar-new2-step010 | sdar-new2-step020 | sdar-new2-step040 |
|----- | ----- | ----- | ----- | ----- | ----- | -----|
| === In-domain Math === | - | - | - | - | - | - |
| math_prm800k_500 | e5f646 | accuracy | gen | 61.00 | 70.00 | 69.20 |
| aime2024 | 59702d | accuracy | gen | 10.00 | 6.67 | 0.00 |
| gsm8k | bd56fc | accuracy | gen | 87.41 | 90.75 | 88.93 |
|  | - | - | - | - | - | - |
| === General Capability === | - | - | - | - | - | - |
| mmlu | - | naive_average | gen | 74.39 | 74.33 | 74.19 |
| IFEval | ea6fc0 | Prompt-level-strict-accuracy | gen | 57.49 | 54.16 | 52.31 |
| hellaswag | 6fd7fe | accuracy | gen | 87.11 | 86.90 | 86.86 |
| openai_humaneval | 56bdbe | humaneval_pass@1 | gen | 74.39 | 78.66 | 79.88 |
| sanitized_mbpp | 4d4ec2 | score | gen | 66.54 | 69.26 | 69.65 |
|  | - | - | - | - | - | - |
| === OOD Transferability === | - | - | - | - | - | - |
| lcb_code_generation | 11f504 | pass@1 | gen | 25.75 | 22.25 | 16.50 |
| GPQA_diamond | 50136a | accuracy | gen | 33.33 | 33.33 | 38.38 |
|  | - | - | - | - | - | - |
| --- MMLU breakdown --- | - | - | - | - | - | - |
| mmlu | - | accuracy | gen | 74.39 | 74.33 | 74.19 |
| mmlu-stem | - | accuracy | gen | 71.02 | 70.69 | 70.29 |
| mmlu-social-science | - | accuracy | gen | 81.94 | 81.74 | 81.59 |
| mmlu-humanities | - | accuracy | gen | 73.44 | 73.63 | 73.72 |
| mmlu-other | - | accuracy | gen | 73.30 | 73.51 | 73.53 |
