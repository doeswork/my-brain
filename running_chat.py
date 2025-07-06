# interactive_nemotron.py
#
# Stream answers from NVIDIA Nemotron-Nano 4B/8B with selectable
# 4-bit, 8-bit, or full-precision FP16 loading.
#
# * 4B model uses the “detailed thinking on/off” prompt style
# * 8B model uses the JSON-style  {'reasoning': True/False} prompt
#   combined with any additional instructions in the same system message.

import os
import torch
import re
from threading import Thread
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

def main() -> None:
    # ------------------------------------------------------------------ #
    # 1) Model & cache directory
    cache_dir = "llm_models"
    os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 2) Model size (4 B vs 8 B)
    size_choice = input("Which model? [4] 4B or [8] 8B? (4/8): ").strip()
    if size_choice == "8":
        model_name = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
    else:
        model_name = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"

    # ------------------------------------------------------------------ #
    # 3) Precision (4-bit / 8-bit / FP16)
    bit_choice = input("Load in [4] 4-bit, [8] 8-bit or [16] 16-bit FP16? (4/8/16): ").strip()
    quant_config = None
    if bit_choice == "4":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif bit_choice == "8":
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
        )
    # else: leave quant_config = None to load full FP16

    # ------------------------------------------------------------------ #
    # 4) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # ------------------------------------------------------------------ #
    # 5) Load model
    print("Loading model (this may take a minute)…")
    if quant_config:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:  # FP16
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.config.pad_token_id = tokenizer.eos_token_id
    print("Model loaded and ready!\n")

    # ------------------------------------------------------------------ #
    # 6) Interactive chat loop
    while True:
        question = input("Your question (or 'exit'): ").strip()
        if question.lower() in ("exit", "quit", "q"):
            break

        resp = input("Thinking mode ([on]/off): ").strip().lower()
        thinking_on = not resp.startswith("off")

        # ------------------------------------------------------------------ #
        # 6a) Build chat messages according to chosen model
        if size_choice == "8":
            # Single system message: reasoning flag + instructions
            system_content = (
                f"{{'reasoning': {thinking_on}}} "
                "Keep answers short (≤ 3 sentences)."
            )
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user",   "content": question},
            ]
        else:
            # 4B: uses detailed thinking inline
            mode = "on" if thinking_on else "off"
            system_content = (
                "Keep answers short (≤ 3 sentences). "
                f"detailed thinking {mode}"
            )
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user",   "content": question},
            ]

        # ------------------------------------------------------------------ #
        # 6b) Convert messages → prompt string via chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

        # ------------------------------------------------------------------ #
        # 6c) Streamed generation setup
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = dict(
            input_ids      = encoded.input_ids,
            attention_mask = encoded.attention_mask,
            max_new_tokens = 1024 if thinking_on else 256,
            temperature    = 0.6   if thinking_on else 1.0,
            top_p          = 0.95  if thinking_on else 1.0,
            do_sample      = thinking_on,
            eos_token_id   = tokenizer.eos_token_id,
            pad_token_id   = tokenizer.eos_token_id,
            repetition_penalty    = 1.1,
            no_repeat_ngram_size  = 2,
            streamer       = streamer,
        )

        # ------------------------------------------------------------------ #
        # 6d) Launch generation in background, stream tokens live
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        print("Assistant: ", end="", flush=True)
        for tok in streamer:
            print(tok, end="", flush=True)
        thread.join()
        print("\n")

if __name__ == "__main__":
    main()
