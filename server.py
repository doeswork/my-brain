# server.py
#
# FastAPI wrapper for Qwen-2.5-Coder-14B in Bits-and-Bytes FP4 (4-bit) on GPU.
# Endpoints:
#   POST /v1/chat/completions   (OpenAI spec)
#   GET  /api/v0/models          (model list for OpenCode)
#   GET  /v1/models              (model list for OpenCode)

import os, time, torch
from typing import List, Dict, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ------------ Chat-Completion schema (OpenAI compatible) ----------------
class Msg(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    model: Optional[str] = None            # ignored, but keeps spec compatibility
    messages: List[Msg]
    max_tokens: Optional[int]    = Field(256, alias="max_tokens")
    temperature: Optional[float] = 0.7
    top_p: Optional[float]       = 0.9
    do_sample: Optional[bool]    = True

class Choice(BaseModel):
    index: int
    message: Msg
    finish_reason: str

class ChatResp(BaseModel):
    id: str
    object: str
    created: int
    choices: List[Choice]
    usage: Dict[str, int]

# ------------------------ Globals & settings ----------------------------
app = FastAPI(title="Local-Qwen-4bit")
CACHE_DIR = "llm_models"
MODEL_ID  = "Qwen/Qwen2.5-Coder-14B-Instruct"
tok = None
model = None

# ------------------------ Startup event ---------------------------
@app.on_event("startup")
def load_model():
    global tok, model
    print("⏳ Startup: loading tokenizer & model …")
    os.makedirs(CACHE_DIR, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True
    )
    tok.pad_token_id = tok.eos_token_id

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit        = True,
        bnb_4bit_quant_type = "fp4",
        bnb_4bit_compute_dtype = torch.float16,
        bnb_4bit_use_double_quant = True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir            = CACHE_DIR,
        quantization_config  = bnb_cfg,
        device_map           = "auto",
        trust_remote_code    = True,
        max_memory           = {0: "11GiB", "cpu": "20GiB"},
    )
    model.eval()
    print("✅ Startup: model ready!")

# ---------------------- Model list endpoints ---------------------------
@app.get("/api/v0/models")
def list_models_v0():
    return {
        "object": "list",
        "data": [ {"id": MODEL_ID, "object": "model"} ]
    }

@app.get("/v1/models")
def list_models():
    # Alias for OpenAI-compatible listing
    return list_models_v0()

# ----------------------------- Inference ------------------------------
@app.post("/v1/chat/completions", response_model=ChatResp)
def completions(req: ChatReq):
    # 1) Build prompt via HF chat template
    text = tok.apply_chat_template(
        [m.dict() for m in req.messages],
        tokenize=False, add_generation_prompt=True
    )
    # 2) Tokenize and move to device
    inputs = tok([text], return_tensors="pt").to(model.device)
    # 3) Generate
    gen_out = model.generate(
        **inputs,
        max_new_tokens = req.max_tokens,
        temperature    = req.temperature,
        top_p          = req.top_p,
        do_sample      = req.do_sample,
        eos_token_id   = tok.eos_token_id,
        pad_token_id   = tok.eos_token_id,
    )
    gen = gen_out[0][ inputs["input_ids"].shape[-1]: ]
    reply = tok.decode(gen, skip_special_tokens=True)

    # 4) Build response
    now = int(time.time())
    usage = {
        "prompt_tokens": inputs["input_ids"].numel(),
        "completion_tokens": gen.numel(),
        "total_tokens": inputs["input_ids"].numel() + gen.numel(),
    }
    choice = Choice(
        index=0,
        message=Msg(role="assistant", content=reply),
        finish_reason="stop"
    )
    return ChatResp(
        id=f"chatcmpl-{now}",
        object="chat.completion",
        created=now,
        choices=[choice],
        usage=usage
    )

# ----------------------------- Runner -------------------------------
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080, workers=1)
