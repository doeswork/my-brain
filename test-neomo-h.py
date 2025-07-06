# simple_nemotron8b.py

import os
import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)

def main():
    # 1) Cache dir & model
    cache_dir = "llm_models"
    os.makedirs(cache_dir, exist_ok=True)
    model_id  = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"

    # 2) Load tokenizer
    print("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # 3) Load model in 8-bit
    print("Loading 8-bit model (takes ~1min)…")
    bnb_cfg = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # 4) Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False
    )

    # 5) User prompts
    reasoning = input("Enable reasoning mode? (yes/no): ").strip().lower().startswith("y")
    question  = input("Enter your question: ").strip()

    # 6) Build messages
    sys_msg = f"detailed thinking on"
    messages = [
        {"role": "system", "content": "detailed thinking on"},
        {"role": "user",   "content": question}
    ]

    # 7) Generate
    print("\nGenerating…\n")
    output = pipe(
        messages,
        max_new_tokens = 512 if reasoning else 128,
        temperature    = 0.6 if reasoning else 0.0,
        top_p          = 0.95 if reasoning else 1.0,
        do_sample      = reasoning
    )

    # 8) Strip reasoning tags if you want only the clean answer
    raw   = output[0]["generated_text"]
    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    print("Message")
    print(messages)
    print("\n=== RAW OUTPUT ===")
    print(raw)
    print("\n=== CLEAN ANSWER ===")
    print(clean)

if __name__ == "__main__":
    main()
