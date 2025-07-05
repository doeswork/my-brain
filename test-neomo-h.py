# test_nemotron_8bit.py

import os
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # 0) Where to cache all HF models
    cache_dir = "llm_models"
    os.makedirs(cache_dir, exist_ok=True)

    # 1) Choose model
    model_id = "nvidia/Nemotron-H-8B-Reasoning-128K"

    # 2) Load tokenizer & model into our local cache_dir, in 8-bit
    print(f"Loading tokenizer for {model_id}…")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading 8-bit quantized model for {model_id}…")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        load_in_8bit=True,             # <<< quantize to 8-bit
        device_map="auto",             # <<< place layers across GPUs/CPU
        trust_remote_code=True
    )
    model.eval()

    # 3) Ask user for reasoning mode
    resp = input("Enable reasoning traces? (yes/no): ").strip().lower()
    reasoning = True if resp.startswith("y") else False

    # 4) Ask user for the question
    question = input("Enter your question: ").strip()

    # 5) Build messages and tokenize via chat template
    # Correctly escape braces to produce "{'reasoning': True/False}"
    system_content = f"{{'reasoning': {reasoning}}}"
    messages = [
        {"role": "system",  "content": system_content},
        {"role": "user",    "content": question},
    ]
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # 6) Choose generation parameters
    max_new_tokens = 1024
    if reasoning:
        do_sample   = True
        temperature = 0.6
        top_p       = 0.95
    else:
        do_sample   = False
        temperature = None
        top_p       = None

    # 7) Generate
    with torch.no_grad():
        outputs = model.generate(
            **tokenized,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            top_k=50,
        )

    # 8) Decode and strip out <think>…</think>
    raw   = tokenizer.decode(outputs[0], skip_special_tokens=True)
    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    print("\n=== Model Response ===")
    print(clean)

if __name__ == "__main__":
    main()
