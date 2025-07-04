# test_llm.py — download into llm_models folder and test DeepSeek-R1 4-bit

import os
import torch
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

# 1) Prepare cache directory
cache_dir = "llm_models"
os.makedirs(cache_dir, exist_ok=True)

# 2) Select the 4-bit DeepSeek-R1-Qwen3-8B model
model_name = "unsloth/DeepSeek-R1-0528-Qwen3-8B-bnb-4bit"

# 3) Configure 4-bit quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# 4) Load tokenizer (with trust_remote_code for any custom repo code)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    trust_remote_code=True,
)

# 5) Load model with 4-bit weights
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

# 6) Build a system prompt (DeepSeek style) with today’s date
current_date = datetime.now().strftime("%B %d, %Y")
system_prompt = (
    "The assistant is DeepSeek-R1, developed by DeepSeek Inc.\n"
    f"Today is {current_date}.\n"
)

# 7) Define your user prompt
user_prompt = "Can you make me a python for loop to go through all the letters of the english alphabet?"

# 8) Combine into the final prompt for the model
full_prompt = f"{system_prompt}\nUser: {user_prompt}\nAssistant:"

# 9) Create a text-generation pipeline
llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    return_full_text=False,
)

# 10) Generate and print the response
response = llm(
    full_prompt,
    max_new_tokens=256,                   # allow up to 512 new tokens
    temperature=0.6,
    top_p=0.95,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,  # stop if model emits this
    pad_token_id=tokenizer.eos_token_id,
    early_stopping=True,                  # return as soon as eos_token_id is hit
    repetition_penalty=1.2,               # discourage token re-use
    no_repeat_ngram_size=3,               # ban repeating any 3-gram
)

print("Assistant:", response[0]["generated_text"])
