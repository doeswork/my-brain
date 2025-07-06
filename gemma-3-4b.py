# test_gemma_3_4b_it.py

import os
import torch
from transformers import pipeline

def main():
    # 1) Where to cache/download models
    cache_dir = "llm_models"
    os.makedirs(cache_dir, exist_ok=True)

    # 2) Model ID
    model_id = "google/gemma-3-4b-it"

    # 3) Initialize the pipeline for image-text-to-text
    #    We’ll use bfloat16 for reduced memory usage on GPU
    print(f"Loading model '{model_id}' into cache_dir='{cache_dir}'…")
    pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        cache_dir=cache_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # 4) Prompt user for a simple text-only test
    print("\nThis model supports multimodal input, but you can test with text alone.")
    question = input("Enter a test prompt: ").strip()

    # 5) Build the required chat‐style message structure
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user",   "content": [{"type": "text", "text": question}]}
    ]

    # 6) Run the pipeline
    print("\nGenerating response…")
    output = pipe(
        text=messages,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9
    )

    # 7) Extract and print the assistant’s reply
    reply = output[0]["generated_text"][-1]["content"]
    print("\n=== Model Reply ===")
    print(reply)

if __name__ == "__main__":
    main()
