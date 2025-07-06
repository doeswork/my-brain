# test_qwen2_coder_4bit.py

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

def main():
    # 1) Directory where models will be cached/downloaded
    cache_dir = "llm_models"
    os.makedirs(cache_dir, exist_ok=True)

    # 2) Model ID on Hugging Face
    model_name = "Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int4"

    # 3) Load tokenizer
    print(f"Loading tokenizer for {model_name}…")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # 4) Configure 4-bit quantization
    print(f"Loading 4-bit quantized model {model_name}… (this may take a minute)")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # 5) Prompt user
    prompt = input("Enter your test prompt: ").strip()

    # 6) Build chat-style input
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user",   "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 7) Tokenize & move to device
    inputs = tokenizer([text], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 8) Generate
    print("\nGenerating…\n")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    # 9) Extract generated portion
    gen = outputs[0][ inputs["input_ids"].shape[-1] : ]
    response = tokenizer.decode(gen, skip_special_tokens=True)

    # 10) Print
    print("\n=== Qwen2.5-Coder 14B 4-bit Response ===")
    print(response)

if __name__ == "__main__":
    main()
