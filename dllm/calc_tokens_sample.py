import json
import numpy as np
from transformers import AutoTokenizer

def main():
    tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/models/SDAR-4B-Chat", trust_remote_code=True)
    
    with open("dllm/train.jsonl", "r") as f:
        lines = f.readlines()
        
    np.random.seed(42)
    sample_indices = np.random.choice(len(lines), 1000, replace=False)
    
    total_tokens = 0
    total_q_tokens = 0
    
    for idx in sample_indices:
        data = json.loads(lines[idx])
        
        # calculate question too just in case
        q = data.get("question", "")
        q_tokens = tokenizer.encode(q, add_special_tokens=False)
        total_q_tokens += len(q_tokens)
        
        # calculate response
        r = data.get("think_response", "")
        r_tokens = tokenizer.encode(r, add_special_tokens=False)
        total_tokens += len(r_tokens)
        
    avg_r = total_tokens / 1000
    avg_q = total_q_tokens / 1000
    print(f"Sample 1000 stats:")
    print(f"Average Question tokens: {avg_q:.2f}")
    print(f"Average Response (think_response) tokens: {avg_r:.2f}")
    print(f"Total entries in dataset: {len(lines)}")

if __name__ == "__main__":
    main()
