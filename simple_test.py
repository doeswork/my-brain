# simple_test.py — load from llm_models/models--nvidia--Llama-3.1-Nemotron-Nano-4B-v1.1

import os
import torch
import transformers

def main():
    # 1) Where your local model lives
    cache_dir = "llm_models"
    local_model_dir = os.path.join(
        cache_dir,
        "models--nvidia--Llama-3.1-Nemotron-Nano-4B-v1.1"
    )
    if not os.path.isdir(local_model_dir):
        raise FileNotFoundError(f"Model directory not found: {local_model_dir}")

    # 2) Model kwargs
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,   # load the model's custom code
    }

    # 3) Load tokenizer from local dir
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        local_model_dir,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # 4) Load model from local dir
    model = transformers.AutoModelForCausalLM.from_pretrained(
        local_model_dir,
        cache_dir=cache_dir,
        **model_kwargs
    )
    model.config.pad_token_id = tokenizer.eos_token_id

    # 5) Create a pipeline with the **model object** (not the string)
    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **model_kwargs
    )

    # 6) Test it
    thinking = "on"   # or "off"
    messages = [
        {"role": "system", "content": f"detailed thinking {thinking}"},
        {"role": "user",   "content": "Solve x*(sin(x)+2)=0"},
    ]
    output = pipe(
        messages,
        max_new_tokens=256,
        temperature=0.6 if thinking=="on" else 1.0,
        top_p=0.95 if thinking=="on" else 1.0,
        do_sample=(thinking=="on"),
    )

    print(output)

if __name__ == "__main__":
    main()
