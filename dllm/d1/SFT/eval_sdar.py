#!/usr/bin/env python3
"""
Evaluation script for SDAR-4B-Chat SFT checkpoints.

Benchmarks:
  Math:    math500, aime
  General: mmlu, ifeval, hellaswag, humaneval
  OOD:     livecode, gpqa

Usage:
  python eval_sdar.py \
    --checkpoint /scratch.global/pan00389/sdar-sft/sdar-sft-test/checkpoint-2500 \
    --benchmarks math500 aime mmlu gpqa \
    --output_dir ./eval_results/ckpt-2500

Notes:
  - ifeval / humaneval / livecode: generates outputs and saves to JSON.
    Run the official scorer separately (see README in each result file).
  - mmlu / hellaswag / gpqa: uses single-forward-pass scoring (fast).
  - math500 / aime: uses diffusion generation + \\boxed{} extraction.
"""

import argparse, hashlib, json, os, re, time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MASK_ID = 151669  # SDAR-4B mask token


# ── Diffusion generation ───────────────────────────────────────────────────────

def _num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=-1, keepdim=True)
    base = mask_num // steps
    rem  = mask_num % steps
    out  = base.repeat(1, steps)
    for i in range(rem.shape[0]):
        out[i, : rem[i].item()] += 1
    return out


@torch.no_grad()
def diffusion_generate(model, prompt_ids, gen_length=512, steps=128,
                        temperature=0.0, device="cuda"):
    bs = prompt_ids.shape[0]
    x = torch.full((bs, prompt_ids.shape[1] + gen_length), MASK_ID,
                   dtype=torch.long, device=device)
    x[:, :prompt_ids.shape[1]] = prompt_ids
    mask_index_all = x == MASK_ID
    num_transfer = _num_transfer_tokens(mask_index_all[:, prompt_ids.shape[1]:], steps)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for i in range(steps):
            cur_mask = x == MASK_ID
            logits = model(x).logits
            if temperature > 0:
                noise = -torch.log(-torch.log(
                    torch.rand_like(logits.float()).clamp(1e-9, 1 - 1e-9)))
                logits = logits.float() + temperature * noise
            x0 = torch.argmax(logits, dim=-1)
            p = F.softmax(logits.float(), dim=-1)
            x0_p = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            x0 = torch.where(cur_mask, x0, x)
            confidence = torch.where(cur_mask, x0_p,
                                     torch.full_like(x0_p, -torch.inf))
            transfer = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(bs):
                n = num_transfer[j, i].item()
                if n > 0:
                    _, idx = torch.topk(confidence[j], k=n)
                    transfer[j, idx] = True
            x[transfer] = x0[transfer]

    return x[:, prompt_ids.shape[1]:]


# ── Multiple-choice scoring (one forward pass per question) ────────────────────

@torch.no_grad()
def score_choices(model, tokenizer, prompt_text, choice_texts, device="cuda"):
    """
    Score each choice by masking its tokens and summing log probs.
    Returns a list of floats (higher = more likely).
    """
    prompt_ids = tokenizer(prompt_text, return_tensors="pt",
                           add_special_tokens=False).input_ids.to(device)
    scores = []
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for choice in choice_texts:
            choice_ids = tokenizer(choice, return_tensors="pt",
                                   add_special_tokens=False).input_ids.to(device)
            x = torch.cat([prompt_ids, choice_ids], dim=1)
            masked = x.clone()
            masked[:, prompt_ids.shape[1]:] = MASK_ID
            logits = model(masked).logits.float()
            log_probs = F.log_softmax(logits, dim=-1)
            score = 0.0
            for pos in range(prompt_ids.shape[1], x.shape[1]):
                score += log_probs[0, pos, x[0, pos]].item()
            scores.append(score)
    return scores


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(tokenizer, user_content):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        add_generation_prompt=True, tokenize=False)


# ── Answer extraction ──────────────────────────────────────────────────────────

def extract_boxed(text):
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    depth, i = 0, idx + len("\\boxed{")
    for c in text[idx + len("\\boxed{"):]:
        if   c == "{": depth += 1
        elif c == "}":
            if depth == 0:
                return text[idx + len("\\boxed{"):i]
            depth -= 1
        i += 1
    return None


# ── MATH-500 ───────────────────────────────────────────────────────────────────

def eval_math500(model, tokenizer, args):
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    correct, results = 0, []
    for item in tqdm(ds, desc="MATH-500"):
        prompt = build_prompt(tokenizer,
            "Solve the following math problem step by step. "
            f"Put your final answer in \\boxed{{}}.\n\n{item['problem']}")
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
        out = diffusion_generate(model, ids, gen_length=args.gen_length,
                                  steps=args.steps, device=args.device)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        pred = extract_boxed(text)
        gold = item["answer"]
        ok = pred is not None and pred.strip() == gold.strip()
        correct += ok
        results.append({"problem": item["problem"], "gold": gold,
                         "pred": pred, "generated": text, "correct": ok})
    return {"accuracy": correct / len(results), "correct": correct,
            "total": len(results), "results": results}


# ── AIME ───────────────────────────────────────────────────────────────────────

def eval_aime(model, tokenizer, args):
    ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    correct, results = 0, []
    for item in tqdm(ds, desc="AIME"):
        prompt = build_prompt(tokenizer,
            "Solve the following competition math problem. "
            "The answer is an integer from 0 to 999. "
            f"Put your final answer in \\boxed{{}}.\n\n{item['problem']}")
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
        out = diffusion_generate(model, ids, gen_length=args.gen_length,
                                  steps=args.steps, device=args.device)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        pred = extract_boxed(text)
        gold = str(item["answer"])
        ok = pred is not None and pred.strip() == gold.strip()
        correct += ok
        results.append({"problem": item["problem"], "gold": gold,
                         "pred": pred, "generated": text, "correct": ok})
    return {"accuracy": correct / max(len(results), 1), "correct": correct,
            "total": len(results), "results": results}


# ── MMLU ───────────────────────────────────────────────────────────────────────

ABCD = ["A", "B", "C", "D"]

def eval_mmlu(model, tokenizer, args):
    ds = load_dataset("cais/mmlu", "all", split="test")
    if args.mmlu_limit > 0:
        ds = ds.select(range(min(args.mmlu_limit, len(ds))))
    correct, results = 0, []
    for item in tqdm(ds, desc="MMLU"):
        choices_text = "\n".join(f"{l}. {c}" for l, c in zip(ABCD, item["choices"]))
        prompt = build_prompt(tokenizer,
            f"Question: {item['question']}\n{choices_text}\n"
            "Answer with only the letter (A, B, C, or D):")
        scores = score_choices(model, tokenizer, prompt,
                               [f" {c}" for c in ABCD], args.device)
        pred_idx = int(np.argmax(scores))
        gold_idx = item["answer"]
        ok = pred_idx == gold_idx
        correct += ok
        results.append({"gold": ABCD[gold_idx], "pred": ABCD[pred_idx], "correct": ok})
    return {"accuracy": correct / len(results), "correct": correct,
            "total": len(results), "results": results}


# ── HellaSwag ──────────────────────────────────────────────────────────────────

def eval_hellaswag(model, tokenizer, args):
    ds = load_dataset("Rowan/hellaswag", split="validation")
    if args.hellaswag_limit > 0:
        ds = ds.select(range(min(args.hellaswag_limit, len(ds))))
    correct, results = 0, []
    for item in tqdm(ds, desc="HellaSwag"):
        endings = item["endings"]
        choices_text = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(endings))
        prompt = build_prompt(tokenizer,
            f"Choose the most likely sentence completion.\n\n"
            f"Context: {item['ctx']}\n\n{choices_text}\n\n"
            "Answer with only the letter (A, B, C, or D):")
        scores = score_choices(model, tokenizer, prompt,
                               [f" {chr(65+i)}" for i in range(len(endings))],
                               args.device)
        pred_idx = int(np.argmax(scores))
        gold_idx = int(item["label"])
        ok = pred_idx == gold_idx
        correct += ok
        results.append({"gold": gold_idx, "pred": pred_idx, "correct": ok})
    return {"accuracy": correct / len(results), "correct": correct,
            "total": len(results), "results": results}


# ── GPQA ───────────────────────────────────────────────────────────────────────

def eval_gpqa(model, tokenizer, args):
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    correct, results = 0, []
    for item in tqdm(ds, desc="GPQA"):
        choices = [item["Correct Answer"],
                   item["Incorrect Answer 1"],
                   item["Incorrect Answer 2"],
                   item["Incorrect Answer 3"]]
        seed = int(hashlib.md5(item["Question"].encode()).hexdigest()[:8], 16) % (2**31)
        rng   = np.random.default_rng(seed)
        order = rng.permutation(4)
        shuffled  = [choices[i] for i in order]
        gold_idx  = int(np.where(order == 0)[0][0])
        choices_text = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(shuffled))
        prompt = build_prompt(tokenizer,
            f"Question: {item['Question']}\n\n{choices_text}\n\n"
            "Answer with only the letter (A, B, C, or D):")
        scores = score_choices(model, tokenizer, prompt,
                               [f" {chr(65+i)}" for i in range(4)], args.device)
        pred_idx = int(np.argmax(scores))
        ok = pred_idx == gold_idx
        correct += ok
        results.append({"gold": gold_idx, "pred": pred_idx, "correct": ok})
    return {"accuracy": correct / len(results), "correct": correct,
            "total": len(results), "results": results}


# ── IFEval (generation only — score separately) ────────────────────────────────

def eval_ifeval(model, tokenizer, args):
    ds = load_dataset("google/IFEval", split="train")
    results = []
    for item in tqdm(ds, desc="IFEval"):
        prompt = build_prompt(tokenizer, item["prompt"])
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
        out = diffusion_generate(model, ids, gen_length=args.gen_length,
                                  steps=args.steps, device=args.device)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        results.append({"prompt": item["prompt"], "generated": text,
                         "key": item["key"],
                         "instruction_id_list": item["instruction_id_list"]})
    return {"total": len(results), "results": results,
            "note": "Score with: python -m instruction_following_eval.evaluation_main "
                    "--input_data=<this_file> --input_response_data=<this_file>"}


# ── HumanEval (generation only — execute separately) ──────────────────────────

def eval_humaneval(model, tokenizer, args):
    ds = load_dataset("openai_humaneval", split="test")
    results = []
    for item in tqdm(ds, desc="HumanEval"):
        prompt = build_prompt(tokenizer,
            "Complete the following Python function. "
            f"Return only the completion, no extra text.\n\n{item['prompt']}")
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
        out = diffusion_generate(model, ids, gen_length=512,
                                  steps=args.steps, device=args.device)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        results.append({"task_id": item["task_id"], "prompt": item["prompt"],
                         "completion": text, "test": item["test"],
                         "entry_point": item["entry_point"]})
    return {"total": len(results), "results": results,
            "note": "Score with: evalplus or human_eval execution harness"}


# ── LiveCodeBench (generation only — execute separately) ──────────────────────

def eval_livecode(model, tokenizer, args):
    ds = load_dataset("livecodebench/code_generation_lite",
                       version_tag="release_v5", split="test")
    results = []
    for item in tqdm(ds, desc="LiveCodeBench"):
        prompt = build_prompt(tokenizer,
            "Solve the following competitive programming problem in Python.\n\n"
            f"{item['question_content']}")
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
        out = diffusion_generate(model, ids, gen_length=1024,
                                  steps=args.steps, device=args.device)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        results.append({"question_id": item.get("question_id", ""),
                         "generated": text})
    return {"total": len(results), "results": results,
            "note": "Score with the official LiveCodeBench execution harness"}


# ── Main ───────────────────────────────────────────────────────────────────────

BENCHMARK_FNS = {
    "math500":   eval_math500,
    "aime":      eval_aime,
    "mmlu":      eval_mmlu,
    "hellaswag": eval_hellaswag,
    "gpqa":      eval_gpqa,
    "ifeval":    eval_ifeval,
    "humaneval": eval_humaneval,
    "livecode":  eval_livecode,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",
                        default="/home/ubuntu/models/SDAR-4B-Chat")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to LoRA checkpoint dir (e.g. checkpoint-2500)")
    parser.add_argument("--benchmarks", nargs="+",
                        default=list(BENCHMARK_FNS.keys()),
                        choices=list(BENCHMARK_FNS.keys()))
    parser.add_argument("--output_dir", default="./eval_results")
    parser.add_argument("--gen_length",       type=int, default=512)
    parser.add_argument("--steps",            type=int, default=128,
                        help="Diffusion denoising steps")
    parser.add_argument("--mmlu_limit",       type=int, default=2000,
                        help="Max MMLU examples (0 = full 14k)")
    parser.add_argument("--hellaswag_limit",  type=int, default=2000,
                        help="Max HellaSwag examples (0 = full 10k)")
    parser.add_argument("--device",           default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_name = Path(args.checkpoint).name

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map=args.device, trust_remote_code=True)

    print(f"Loading LoRA: {args.checkpoint}")
    model = PeftModel.from_pretrained(model, args.checkpoint)
    model = model.merge_and_unload()
    model.eval()
    print(f"Ready. Benchmarks: {args.benchmarks}\n")

    summary = {}
    for bm in args.benchmarks:
        print(f"\n{'='*50}  {bm}  {'='*50}")
        t0  = time.time()
        res = BENCHMARK_FNS[bm](model, tokenizer, args)
        res["elapsed_sec"] = round(time.time() - t0, 1)

        out_path = os.path.join(args.output_dir, f"{ckpt_name}_{bm}.json")
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

        acc = res.get("accuracy")
        print(f"  {bm}: accuracy={acc:.4f}  ({res['elapsed_sec']}s)  -> {out_path}"
              if acc is not None else
              f"  {bm}: generations saved -> {out_path}")
        summary[bm] = {k: v for k, v in res.items() if k != "results"}

    summary_path = os.path.join(args.output_dir, f"{ckpt_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}  SUMMARY  {'='*50}")
    for bm, s in summary.items():
        acc = s.get("accuracy")
        print(f"  {bm:12s}: {acc:.4f}" if acc is not None else f"  {bm:12s}: see JSON")
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()