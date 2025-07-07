# server.py
#
# FastAPI wrapper around Qwen-2.5-Coder-7B-Instruct in 4-bit on GPU.
# Now supports both streaming (SSE) and non-streaming responses.
# Endpoints:
#   GET  /v1/models
#   POST /v1/chat/completions

import time
import torch
import uvicorn
from threading import Thread
from typing import List, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

# ——— Configuration ———
CACHE_DIR = "llm_models"
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"

# ——— API Schemas ———
class Msg(BaseModel):
    role: str
    content: str

    @field_validator("content", mode="before")
    def coerce_content(cls, v):
        if isinstance(v, list):
            return "".join(part.get("text", str(part)) for part in v)
        return v

class ChatReq(BaseModel):
    model: Optional[str] = None
    messages: List[Msg]
    max_tokens: Optional[int] = Field(1042, alias="max_tokens")
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    do_sample: Optional[bool] = True
    stream: Optional[bool] = False  # The crucial flag to detect streaming requests

class Choice(BaseModel):
    index: int
    message: Msg
    finish_reason: Optional[str] = None

class ChatResp(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[Choice]
    usage: Optional[Dict[str, int]] = None

class DeltaMsg(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class StreamingChoice(BaseModel):
    index: int
    delta: DeltaMsg
    finish_reason: Optional[str] = None

class StreamingChatResp(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamingChoice]

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
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
    max_memory={0: "8GiB", "cpu": "16GiB"},
)
model.eval()
print("✅ Model ready!")

# ——— FastAPI app & endpoints ———
app = FastAPI(title="Local-Qwen-7B-4bit")

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(data=[ModelCard(id=MODEL_ID)])

async def stream_generator(req: ChatReq, chat_id: str):
    """Generator function that yields SSE events."""
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    
    prompt = tok.apply_chat_template(
        [m.model_dump() for m in req.messages],
        tokenize=False, add_generation_prompt=True
    )
    inputs = tok([prompt], return_tensors="pt").to(model.device)

    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        do_sample=req.do_sample,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    
    # Run generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Yield chunks as they become available
    for new_text in streamer:
        chunk = StreamingChatResp(
            id=chat_id,
            model=MODEL_ID,
            choices=[StreamingChoice(index=0, delta=DeltaMsg(content=new_text))]
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    # Signal the end of the stream
    end_chunk = StreamingChatResp(
        id=chat_id,
        model=MODEL_ID,
        choices=[StreamingChoice(index=0, delta=DeltaMsg(), finish_reason="stop")]
    )
    yield f"data: {end_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def completions(req: ChatReq):
    chat_id = f"chatcmpl-{int(time.time())}"
    
    if req.stream:
        # Handle streaming requests
        return StreamingResponse(
            stream_generator(req, chat_id),
            media_type="text/event-stream"
        )
    else:
        # Handle non-streaming requests
        prompt = tok.apply_chat_template(
            [m.model_dump() for m in req.messages],
            tokenize=False, add_generation_prompt=True
        )
        inputs = tok([prompt], return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=req.max_tokens)
        gen = out[0][inputs["input_ids"].shape[-1]:]
        reply = tok.decode(gen, skip_special_tokens=True)
        
        return ChatResp(
            id=chat_id,
            choices=[Choice(index=0, message=Msg(role="assistant", content=reply), finish_reason="stop")]
        )

# ——— Run with uvicorn ———
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=1)