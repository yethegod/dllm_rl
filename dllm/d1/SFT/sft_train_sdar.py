import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import json
from peft import LoraConfig, get_peft_model, TaskType
import os
from sft_trainer import dLLMTrainer, dLLMSFTDataset, dLLMDataCollator, SYSTEM_PROMPT
from tqdm import tqdm
import random
import numpy as np

SDAR_MASK_TOKEN_ID = 151669


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="/users/8/pan00389/dllm/models/SDAR-4B-Chat"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument(
        "--output_dir", type=str, default="/users/8/pan00389/dllm/checkpoints/sdar-sft"
    )
    parser.add_argument("--job_name", type=str, default="sdar-4b-sft")
    parser.add_argument("--train_data", type=str, default="/users/8/pan00389/dllm/train.jsonl")
    parser.add_argument("--save_epochs", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=None, help="Override save_epochs with a fixed step interval")
    parser.add_argument("--debugging", action="store_true")
    return parser.parse_args()


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, padding_side="right", trust_remote_code=True, use_fast=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model = model.to(torch.bfloat16)
    model.print_trainable_parameters()

    return tokenizer, model


def preprocess_sdar(data, tokenizer, max_length, test_split=0.01):
    preprocessed_data = []
    for item in tqdm(data, desc="Preprocessing dataset"):
        question = SYSTEM_PROMPT + "\n\n" + item["question"]
        trajectory = f"<reasoning>{item['no_think_response']}</reasoning>\n<answer>{item['golden_answer']}</answer>"
        prompt = [{"role": "user", "content": question}]
        response = [{"role": "assistant", "content": trajectory}]
        inputs = tokenizer.apply_chat_template(prompt + response, tokenize=False)
        prompt_str = tokenizer.apply_chat_template(prompt, tokenize=False) + "\n"
        tokenized_input = tokenizer(
            inputs, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length"
        ).input_ids.squeeze(0)
        tokenized_prompt = tokenizer(prompt_str, return_tensors="pt", truncation=True, max_length=max_length)
        preprocessed_data.append({
            "input_ids": tokenized_input,
            "prompt_lengths": tokenized_prompt.attention_mask.sum(-1),
        })
    random.shuffle(preprocessed_data)
    split = int(len(preprocessed_data) * test_split)
    return preprocessed_data[split:], preprocessed_data[:split]


def load_data(args, tokenizer):
    import hashlib, pickle, pathlib, time
    cache_key = hashlib.md5(f"{args.train_data}_{args.max_length}".encode()).hexdigest()[:8]
    cache_path = pathlib.Path(args.train_data).parent / f".cache_sdar_{cache_key}.pkl"
    done_path = cache_path.with_suffix(".done")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank == 0 and not done_path.exists():
        with open(args.train_data) as f:
            data = [json.loads(line) for line in f]
        train_data, eval_data = preprocess_sdar(data, tokenizer, args.max_length)
        with open(cache_path, "wb") as f:
            pickle.dump((train_data, eval_data), f)
        done_path.touch()
        print(f"Saved preprocessed data to cache: {cache_path}")
    else:
        while not done_path.exists():
            time.sleep(5)

    print(f"Loading preprocessed data from cache: {cache_path}")
    with open(cache_path, "rb") as f:
        train_data, eval_data = pickle.load(f)

    print("Train data length: ", len(train_data))
    print("Eval data length: ", len(eval_data))
    train_dataset = dLLMSFTDataset(train_data, tokenizer, args.max_length)
    eval_dataset = dLLMSFTDataset(eval_data, tokenizer, args.max_length, eval=True)
    return train_dataset, eval_dataset


def train_model(args, tokenizer, model):
    train_dataset, eval_dataset = load_data(args, tokenizer)

    import torch.distributed as dist
    world_size = dist.get_world_size() if dist.is_initialized() else max(1, torch.cuda.device_count())
    steps_per_epoch = max(1, len(train_dataset) // (args.batch_size * args.grad_accum_steps * world_size))
    if args.save_steps is not None:
        save_steps = args.save_steps
    else:
        save_steps = steps_per_epoch * args.save_epochs
    total_steps = steps_per_epoch * args.num_epochs
    print(f"steps/epoch: {steps_per_epoch}, total steps: {total_steps}, saving every {save_steps} steps")

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.job_name),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        eval_strategy="steps",
        eval_steps=save_steps,
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=5,
        learning_rate=args.learning_rate,
        load_best_model_at_end=True,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        report_to="wandb" if not args.debugging else "none",
        remove_unused_columns=False,
    )

    trainer = dLLMTrainer(
        model=model,
        args=training_args,
        data_collator=dLLMDataCollator(
            tokenizer=tokenizer,
            mask_token_id=SDAR_MASK_TOKEN_ID,
            max_length=args.max_length,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Auto-resume from latest checkpoint if one exists
    checkpoint = None
    run_dir = os.path.join(args.output_dir, args.job_name)
    if os.path.isdir(run_dir):
        ckpts = sorted(
            [d for d in os.listdir(run_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )
        if ckpts:
            checkpoint = os.path.join(run_dir, ckpts[-1])
            print(f"Resuming from checkpoint: {checkpoint}")

    trainer.train(resume_from_checkpoint=checkpoint)


if __name__ == "__main__":
    init_seed(42)
    args = parse_args()
    tokenizer, model = load_model_and_tokenizer(args)
    train_model(args, tokenizer, model)
