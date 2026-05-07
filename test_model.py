import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CKPT = '/home/ubuntu/uq/checkpoints/checkpoint-100'

print(f'Loading {CKPT}...')
tokenizer = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(CKPT, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda')

messages = [{'role': 'user', 'content': 'What is 2 + 2? Think step by step.'}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors='pt').to('cuda')

print('Generating...')
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.95)

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print('=== Response ===')
print(response)
