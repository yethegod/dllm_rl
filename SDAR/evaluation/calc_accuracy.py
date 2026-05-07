#!/usr/bin/env python3
"""Calculate accuracy for finished OpenCompass prediction files across all checkpoints."""
import json
import os
import re
import sys
from collections import defaultdict

BASE_DIR = (
    "/lambda/nfs/agent/SDAR/evaluation/opencompass/outputs/"
    "eval-sdar-ckpts-050-120/20260506_183439/predictions"
)

# ── extraction helpers ────────────────────────────────────────────────────────

def extract_boxed(text):
    matches = list(re.finditer(r'\\boxed\{', text))
    if not matches:
        return None
    start = matches[-1].end()
    depth, i = 1, start
    while i < len(text) and depth:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    return text[start:i - 1].strip() if depth == 0 else None


def normalize_number(s):
    if s is None:
        return None
    s = s.strip().replace(',', '').replace(' ', '')
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, OverflowError):
        return s.lower()


def extract_gsm8k_gold(gold):
    m = re.search(r'####\s*([\d,\.\-]+)', gold)
    return normalize_number(m.group(1)) if m else normalize_number(gold)


def extract_mmlu_pred(text):
    """Extract first A/B/C/D from prediction, stripping special tokens."""
    text = re.sub(r'<\|.*?\|>', '', text).strip()
    m = re.match(r'\s*([ABCD])', text)
    return m.group(1).upper() if m else None


def extract_mathbench_gold_letter(gold):
    parts = gold.split('--')
    return parts[1].strip().upper() if len(parts) >= 2 else gold.strip().upper()


def extract_choice(text):
    for pat in [
        r'所以答案为选项\s*([ABCD])', r'答案为选项\s*([ABCD])',
        r'答案是\s*([ABCD])', r'答案为\s*([ABCD])',
        r'答案：\s*([ABCD])', r'answer is\s*\(?([ABCD])\)?',
        r'answer:\s*([ABCD])', r'\\boxed\{([ABCD])\}',
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    hits = re.findall(r'\b([ABCD])\b', text)
    return hits[-1].upper() if hits else None


# ── per-dataset scorers ───────────────────────────────────────────────────────

def score_gsm8k(entry):
    pred = extract_boxed(entry.get('prediction', ''))
    gold = extract_gsm8k_gold(entry.get('gold', ''))
    return normalize_number(pred) == gold


def score_math(entry):
    pred = extract_boxed(entry.get('prediction', ''))
    gold = entry.get('gold', '').strip()
    pn, gn = normalize_number(pred), normalize_number(gold)
    if pn is not None and gn is not None and pn == gn:
        return True
    gold_inner = extract_boxed(gold) or gold
    return (pred or '').strip() == gold_inner.strip()


def score_aime(entry):
    pred = extract_boxed(entry.get('prediction', ''))
    gold = entry.get('gold', '').strip()
    return normalize_number(pred) == normalize_number(gold)


def score_mmlu(entry):
    pred = extract_mmlu_pred(entry.get('prediction', ''))
    gold = entry.get('gold', '').strip().upper()
    return pred == gold


def score_mathbench_choice(entry):
    pred = extract_choice(entry.get('prediction', ''))
    gold = extract_mathbench_gold_letter(entry.get('gold', ''))
    return pred == gold


def get_scorer_and_group(filename):
    """Return (scorer_fn, dataset_key) for a filename."""
    if 'gsm8k' in filename:
        return score_gsm8k, 'gsm8k'
    if 'math_prm800k' in filename or 'math500' in filename:
        return score_math, 'math_prm800k_500'
    if 'aime' in filename:
        return score_aime, 'aime2024'
    if 'lukaemon_mmlu' in filename:
        # strip shard number to aggregate all MMLU sub-datasets together
        return score_mmlu, 'mmlu'
    if 'mathbench' in filename and 'single_choice' in filename:
        # group by mathbench sub-type (college/high/middle × cn/en)
        m = re.match(r'(mathbench-[a-z]+-single_choice_[a-z]+)_\d+\.json', filename)
        key = m.group(1) if m else filename.rsplit('_', 1)[0]
        return score_mathbench_choice, key
    return None, None


# ── main ─────────────────────────────────────────────────────────────────────

def score_checkpoint(ckpt_dir):
    """Return dict: dataset_key -> (correct, total)."""
    stats = defaultdict(lambda: [0, 0])
    for fname in sorted(os.listdir(ckpt_dir)):
        if not fname.endswith('.json'):
            continue
        scorer, key = get_scorer_and_group(fname)
        if scorer is None:
            continue
        data = json.load(open(os.path.join(ckpt_dir, fname)))
        for entry in data.values():
            stats[key][1] += 1
            if scorer(entry):
                stats[key][0] += 1
    return stats


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else BASE_DIR

    checkpoints = sorted(
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and not d.startswith('.')
    )

    all_results = {}  # ckpt -> {dataset -> (c, t)}
    all_datasets = set()

    print("Scoring checkpoints...")
    for ckpt in checkpoints:
        ckpt_path = os.path.join(base, ckpt)
        stats = score_checkpoint(ckpt_path)
        all_results[ckpt] = stats
        all_datasets.update(stats.keys())
        print(f"  {ckpt}: {len(stats)} datasets scored")

    datasets = sorted(all_datasets)

    # ── per-checkpoint shard detail ──
    print("\n" + "=" * 80)
    print("PER-CHECKPOINT SUMMARY")
    print("=" * 80)

    col_w = 10
    header = f"{'dataset':<40}" + "".join(f"{c[-6:]:>{col_w}}" for c in checkpoints)
    print(header)
    print("-" * len(header))

    for ds in datasets:
        row = f"{ds:<40}"
        for ckpt in checkpoints:
            s = all_results[ckpt].get(ds)
            if s and s[1] > 0:
                row += f"{s[0]/s[1]*100:>{col_w}.1f}"
            else:
                row += f"{'—':>{col_w}}"
        print(row)

    # ── overall per-checkpoint ──
    print("\n" + "=" * 80)
    print("OVERALL PER CHECKPOINT  (correct / total  =  acc%)")
    print("=" * 80)
    for ckpt in checkpoints:
        total_c = sum(v[0] for v in all_results[ckpt].values())
        total_t = sum(v[1] for v in all_results[ckpt].values())
        parts = []
        for ds in datasets:
            s = all_results[ckpt].get(ds)
            if s and s[1] > 0:
                parts.append(f"{ds.split('_')[0]}: {s[0]/s[1]*100:.1f}%")
        print(f"\n  {ckpt}")
        for ds in datasets:
            s = all_results[ckpt].get(ds)
            if s and s[1] > 0:
                acc = s[0] / s[1] * 100
                missing = "(partial)" if s[1] < max(
                    (all_results[c].get(ds, [0, 0])[1] for c in checkpoints), default=0
                ) else ""
                print(f"    {ds:<40} {s[0]:>5}/{s[1]:<5} = {acc:5.1f}%  {missing}")


if __name__ == '__main__':
    main()
