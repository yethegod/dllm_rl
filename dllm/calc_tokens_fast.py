import json
import numpy as np
from transformers import AutoTokenizer

def main():
    tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/models/SDAR-4B-Chat", trust_remote_code=True, use_fast=True)
    
    with open("dllm/train.jsonl", "r") as f:
        lines = f.readlines()
        
    responses = []
    for line in lines:
        data = json.loads(line)
        responses.append(data.get("think_response", ""))
        
    batch_size = 1000
    total_tokens = 0
    total_len = len(responses)
    
    for i in range(0, total_len, batch_size):
        batch = responses[i:i+batch_size]
        # fast batch tokenization
        tokenized = tokenizer(batch, add_special_tokens=False)["input_ids"]
        for t in tokenized:
            total_tokens += len(t)
        print(f"Processed {i+len(batch)}/{total_len}")
        
    avg = total_tokens / total_len
    print(f"Average think_response tokens: {avg:.2f}")

if __name__ == "__main__":
    main()
