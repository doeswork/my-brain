# server.py
#
# FastAPI wrapper around Qwen-2.5-Coder-14B in 4-bit on GPU (fp4 on GPU, nf4 if offloaded to CPU).
# Endpoints:
#   GET  /v1/models
#   POST /v1/chat/completions

import time
import torch
from typing import List, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
import uvicorn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ——— Configuration ———
CACHE_DIR = "llm_models"
MODEL_ID  = "Qwen/Qwen2.5-Coder-14B-Instruct"

# ——— API Schemas ———
class Msg(BaseModel):
    role: str
    content: str

    @field_validator("content", mode="before")
    def coerce_content(cls, v):
        if isinstance(v, list):
            return "".join(p.get("text", str(p)) for p in v)
        return v

class ChatReq(BaseModel):
    model: Optional[str] = None
    messages: List[Msg]
    max_tokens: Optional[int] = Field(256, alias="max_tokens")
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    do_sample: Optional[bool] = True

class Choice(BaseModel):
    index: int
    message: Msg
    finish_reason: str

class ChatResp(BaseModel):
    id: str
    object: str
    created: int
    choices: List[Choice]
    usage: Dict[str,int]

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "user"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]

# ——— Load tokenizer & model exactly once ———
print("⏳ Loading tokenizer…")
tok = AutoTokenizer.from_pretrained(
    MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True
)
tok.pad_token_id = tok.eos_token_id

print("⏳ Loading 4-bit model…")
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit               = True,
    bnb_4bit_quant_type        = "nf4",        # nf4 so CPU offload works
    bnb_4bit_compute_dtype     = torch.float16,
    bnb_4bit_use_double_quant  = True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir           = CACHE_DIR,
    quantization_config = bnb_cfg,
    device_map          = "auto",
    trust_remote_code   = True,
    max_memory          = {0: "11GiB", "cpu": "20GiB"},
)
model.eval()
print("✅ Model ready!")

# ——— FastAPI app & endpoints ———
app = FastAPI(title="Local-Qwen-4bit")

@app.get("/v1/models", response_model=ModelList)
def list_models():
    return ModelList(data=[ModelCard(id=MODEL_ID)])

@app.post("/v1/chat/completions", response_model=ChatResp)
def completions(req: ChatReq):
    # 1) build prompt
    prompt = tok.apply_chat_template(
        [m.model_dump() for m in req.messages],
        tokenize=False, add_generation_prompt=True
    )

    # 2) tokenize and send to device
    inputs = tok([prompt], return_tensors="pt").to(model.device)

    # 3) generate
    out = model.generate(
        **inputs,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        do_sample=req.do_sample,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )

    # 4) slice off prompt
    gen = out[0][inputs["input_ids"].shape[-1]:]
    reply = tok.decode(gen, skip_special_tokens=True)

    # 5) wrap in OpenAI-style response
    now = int(time.time())
    usage = {
        "prompt_tokens":     inputs["input_ids"].numel(),
        "completion_tokens": gen.numel(),
        "total_tokens":      inputs["input_ids"].numel() + gen.numel(),
    }
    choice = Choice(
        index=0,
        message=Msg(role="assistant", content=reply),
        finish_reason="stop",
    )
    return ChatResp(
        id=f"chatcmpl-{now}",
        object="chat.completion",
        created=now,
        choices=[choice],
        usage=usage,
    )

# ——— Run with uvicorn ———
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=1)
