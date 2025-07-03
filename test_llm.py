import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Directory where models will be cached/downloaded
cache_dir = "llm_models"
os.makedirs(cache_dir, exist_ok=True)

# Choose the model you want
model_name = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"

# Load tokenizer and model, pointing cache_dir to llm_models/
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    torch_dtype="auto",
    device_map="auto"
)

# Define your prompt
prompt = "What color is the sky?"

# Set up the pipeline
llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# Generate a response
response = llm(
    prompt,
    max_new_tokens=100,
    temperature=0.3,
    top_p=0.9,
    top_k=50,
    do_sample=True,
)

print("Assistant:", response[0]["generated_text"])
