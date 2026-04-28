import json
import numpy as np
from transformers import AutoTokenizer
import tqdm

def main():
    tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/models/SDAR-4B-Chat", trust_remote_code=True)
    
    think_lengths = []
    
    with open("dllm/train.jsonl", "r") as f:
        lines = f.readlines()
        
    for line in tqdm.tqdm(lines):
        data = json.loads(line)
        response = data.get("think_response", "")
        # Calculate tokens
        tokens = tokenizer.encode(response, add_special_tokens=False)
        think_lengths.append(len(tokens))
        
    avg = np.mean(think_lengths)
    print(f"Average think_response tokens: {avg:.2f}")
    
if __name__ == "__main__":
    main()
