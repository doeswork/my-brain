# test_llm.py — download into llm_models folder and test Nemotron-Nano-4B in 8-bit

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

# 1) Prepare cache directory
cache_dir = "llm_models"
os.makedirs(cache_dir, exist_ok=True)

# 2) Choose your model
model_name = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"

# 3) Configure 8-bit quantization
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,                # enable 8-bit weights
    bnb_8bit_compute_dtype=torch.float16,  # do compute in FP16
)

# 4) Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    trust_remote_code=True,
)
tokenizer.pad_token_id = tokenizer.eos_token_id

# 5) Load model in 8-bit
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    quantization_config=quant_config,
    device_map="auto",               # requires accelerate
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model.config.pad_token_id = tokenizer.eos_token_id

# 6) Define your prompt
prompt = "What color is the sky?"

# 7) Set up the pipeline
llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    return_full_text=False,
)

# 8) Generate a response
response = llm(
    prompt,
    max_new_tokens=256,
    temperature=0.3,
    top_p=0.9,
    top_k=50,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.1,
    no_repeat_ngram_size=2,
)

print("Assistant:", response[0]["generated_text"])
