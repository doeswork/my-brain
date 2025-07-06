# test_qwen2.5_int4.py
import os, torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

CACHE_DIR  = "llm_models"
MODEL_ID   = "Qwen/Qwen2.5-Coder-14B-Instruct"

def load_model():
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"⏳ Loading tokenizer for {MODEL_ID} …")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print("⏳ Loading 4-bit quantized model (FP4 on CUDA) …")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit       = True,
        bnb_4bit_quant_type= "fp4",          # fp4 only on GPU
        bnb_4bit_compute_dtype = torch.float16,
        bnb_4bit_use_double_quant = True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        quantization_config = bnb_cfg,
        device_map = "auto",                 # place layers on the GPU
        trust_remote_code = True,
        max_memory = {0: "11GiB", "cpu": "20GiB"},
    )
    model.eval()

    # sanity check
    first = next(model.parameters())
    print("✅ Model ready | dtype:", first.dtype,
          "| loaded_in_4bit:", getattr(model, "is_loaded_in_4bit", False))
    return tokenizer, model

def main():
    tok, mdl = load_model()
    while True:
        prompt = input("\nYour prompt (or q): ").strip()
        if prompt.lower() in {"q", "quit", "exit"}:
            break

        msgs = [
            {"role": "system", "content": "You are Qwen, a helpful coding assistant."},
            {"role": "user",   "content": prompt}
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok([text], return_tensors="pt").to(mdl.device)

        print("\n⏳ Generating …\n")
        t0 = time.time()
        out = mdl.generate(
            **inputs,
            max_new_tokens = 256,
            temperature    = 0.7,
            top_p          = 0.9,
            do_sample      = True,
            eos_token_id   = tok.eos_token_id,
            pad_token_id   = tok.eos_token_id,
        )
        gen = out[0][inputs["input_ids"].shape[-1]:]
        print(tok.decode(gen, skip_special_tokens=True))
        print(f"\n⏱️  Took {time.time()-t0:.1f}s\n")

if __name__ == "__main__":
    main()
