# test_model_name_4bit.py

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def main():
    # 1) Where to cache/download the model
    cache_dir = "llm_models"
    os.makedirs(cache_dir, exist_ok=True)

    # 2) Model ID on Hugging Face
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

    # 3) Load tokenizer
    print(f"⏳ Loading tokenizer for {model_name}…")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # 4) Configure 4-bit quantization
    print(f"⏳ Loading 4-bit quantized model {model_name}… (this may take a minute)")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",        # or "fp4" for fp4 kernels
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("✅ Model ready!")

if __name__ == "__main__":
    main()
