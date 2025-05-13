#pip install transformers accelerate
#pip install torch --index-url https://download.pytorch.org/whl/cu121
#login to Hugging Face
    # pip install huggingface_hub
    # huggingface-cli login

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

#model_name = "meta-llama/Llama-3.2-3b-instruct"
model_name = "meta-llama/Llama-3.2-1b-instruct"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

# Define prompt
prompt = "What color is the sky?"

# Set up inference pipeline
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate response
response = llm(
    prompt,
    max_new_tokens=100,
    temperature=0.3,    # lower temp → more deterministic
    top_p=0.9,
    top_k=50,
    do_sample=True,
)

# Print output
print("Assistant:", response[0]["generated_text"])