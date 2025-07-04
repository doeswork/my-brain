# test_llm_loop.py — interactive streaming for Nemotron-Nano-4B

import os
import torch
from threading import Thread
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

def main():
    # 1) Prepare cache
    cache_dir = "llm_models"
    os.makedirs(cache_dir, exist_ok=True)
    model_name = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"

    # 2) Choose precision
    choice = input("Load model in [8] 8-bit or [16] 16-bit FP16? (8/16): ").strip()
    use_8bit = (choice == "8")
    quant_config = None
    if use_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
        )

    # 3) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # 4) Load model once
    print("Loading model (this may take a minute)...")
    if use_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.config.pad_token_id = tokenizer.eos_token_id
    print("Model loaded and ready!\n")

    # 5) Interactive loop
    while True:
        question = input("Your question (or 'exit'): ").strip()
        if question.lower() in ("exit", "quit", "q"):
            break

        thinking = input("Thinking mode ([on]/off): ").strip().lower()
        thinking = "on" if thinking not in ("off",) else "off"

        # Build the full prompt
        messages = [
            {
              "role": "system",
              "content": (
                  "You are Nemotron-Nano. Keep answers short (≤ 3 sentences). "
                  f"detailed thinking {thinking}"
              )
            },
            {"role": "user", "content": question},
        ]
        
        # 2) Turn them into token IDs
        # The 'encoded' variable is the input_ids tensor directly.
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        # Set up streaming
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # Launch generate in background
        gen_kwargs = dict(
            input_ids=input_ids,
            # attention_mask is not needed here; model handles it
            max_new_tokens=512,
            temperature=0.6 if thinking=="on" else 1.0,
            top_p=0.95 if thinking=="on" else 1.0,
            do_sample=(thinking=="on"),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=2,
            streamer=streamer,
        )
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        # Stream tokens as they come
        print("Assistant: ", end="", flush=True)
        for token in streamer:
            print(token, end="", flush=True)
        thread.join()
        print("\n")

if __name__ == "__main__":
    main()