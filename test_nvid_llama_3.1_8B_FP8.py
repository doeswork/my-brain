# test_nemotron_nano_8b.py — download into llm_models/ and test Nemotron-Nano-8B in 8-bit

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

def main():
    # 1) Prepare cache directory
    cache_dir = "llm_models"
    os.makedirs(cache_dir, exist_ok=True)

    # 2) Model identifier
    model_name = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"

    # 3) 8-bit quantization config (fits in ~12 GB VRAM)
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
    )

    # 4) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # 5) Load model with 8-bit weights
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.eos_token_id

    # 6) Build two pipelines: one for reasoning on, one for off
    # pipe_on = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     device_map="auto",
    #     return_full_text=False,
    # )

    pipe_off = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        return_full_text=False,
        do_sample=False,  # greedy decoding when reasoning is off
    )

    # 7) Prepare prompts
    # system_on  = {"role":"system", "content":"detailed thinking on"}
    system_off = {"role":"system", "content":"detailed thinking off"}
    user       = {"role":"user",   "content":"What is the color of the sky?"}

    # # 8) Run “Reasoning On”
    # print("\n=== Reasoning ON ===")
    # out_on = pipe_on([system_on, user], max_new_tokens=256, temperature=0.6, top_p=0.95)
    # print(out_on[0]["generated_text"])

    # 9) Run “Reasoning Off”
    print("\n=== Reasoning OFF ===")
    out_off = pipe_off([system_off, user], max_new_tokens=256)
    print(out_off[0]["generated_text"])

if __name__ == "__main__":
    main()
