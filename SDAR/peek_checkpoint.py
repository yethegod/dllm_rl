"""Peek at a checkpoint: greedy token-by-token gen using training model in eval mode."""
import sys, json, os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "/lambda/nfs/agent/checkpoints/sdar_4b_nothink/checkpoint-20"
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
MASK_ID = 151669

QUESTION = "What is 2 + 2? Please reason step by step, and put your final answer within \\boxed{}."

print(f"Loading: {CHECKPOINT}")

cfg_path = os.path.join(CHECKPOINT, "config.json")
with open(cfg_path) as f:
    cfg = json.load(f)
orig_attn = cfg.get("attn_implementation", "flex_attention")
cfg["attn_implementation"] = "eager"
cfg["use_cache"] = False
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)

try:
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
    model.eval()

    messages = tokenizer.apply_chat_template(
        [{"role": "user", "content": QUESTION}],
        add_generation_prompt=True, tokenize=False,
    )
    prompt_ids = tokenizer(messages, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    print(f"Prompt ({prompt_ids.shape[1]} tokens): {QUESTION}\n")

    # Greedy autoregressive generation (no block diffusion), just to peek at the model's logits
    generated = []
    cur_ids = prompt_ids.clone()
    with torch.no_grad():
        for _ in range(MAX_TOKENS):
            out = model(cur_ids)
            next_logit = out.logits[0, -1]  # last position
            next_tok = next_logit.argmax().item()
            if next_tok == tokenizer.eos_token_id:
                break
            generated.append(next_tok)
            cur_ids = torch.cat([cur_ids, torch.tensor([[next_tok]], device=model.device)], dim=1)

    print(f"Raw first 20 tokens: {generated[:20]}")
    print("\n=== Greedy Generation ===")
    print(tokenizer.decode(generated, skip_special_tokens=False))
    print("=========================")

finally:
    cfg["attn_implementation"] = orig_attn
    cfg["use_cache"] = False
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
