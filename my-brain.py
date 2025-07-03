# my_brain.py

import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document as LC_Document

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# where to stash all HF models
cache_dir = "llm_models"
os.makedirs(cache_dir, exist_ok=True)


# 1) Load and split your .txt docs
def build_vectorstore(context_dir: str, persist_dir: str = "chroma_db"):
    docs = []
    for fn in os.listdir(context_dir):
        if fn.lower().endswith(".txt"):
            path = os.path.join(context_dir, fn)
            text = open(path, encoding="utf-8").read()
            docs.append(LC_Document(page_content=text, metadata={"source": fn}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=persist_dir
    )
    return db


# 2) Load your LLM once
def load_llm(
    model_name: str = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1",
    cache_dir: str = cache_dir
):
    # load tokenizer & model into our local cache_dir
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype="auto",
        device_map="auto"
    )

    # return a generation pipeline
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False
    )


# 3) Answer a query (more conversational)
def answer_query(db, llm, query: str, thinking: str, k: int = 3):
    # retrieve top-k context chunks
    results = db.similarity_search(query, k=k)
    context = "\n\n".join(r.page_content for r in results)

    # build a conversational prompt
    sys_prompt = (
        "You are a friendly assistant. "
        "Use a natural, conversational tone when answering. Make your response short and simple."
    )
    user_prompt = (
        f"Here’s some context:\n{context}\n\n"
        f"User’s question: {query}\n"
        "Assistant:"
    )

    # assemble messages for Nemotron
    messages = [
        {"role": "system",  "content": sys_prompt},
        {"role": "system",  "content": f"detailed thinking {thinking}"},
        {"role": "user",    "content": user_prompt},
    ]

    # choose sampling params
    do_sample = (thinking == "on")
    temp     = 0.7 if do_sample else 0.0
    top_p    = 0.9 if do_sample else 1.0

    out = llm(
        messages,
        max_new_tokens=1024,
        temperature=temp,
        top_p=top_p,
        top_k=50,
        do_sample=do_sample,
    )

    print(out[0]["generated_text"].strip())

if __name__ == "__main__":
    # build (or reuse) the vectorstore
    VDB = build_vectorstore(context_dir="context")

    # load the LLM pipeline
    LLM = load_llm()

    # ask user whether to enable detailed thinking
    resp = input("Enable detailed thinking mode? (yes/no): ").strip().lower()
    thinking = "on" if resp.startswith("y") else "off"

    # ask for the actual question
    query = input("Enter your question: ").strip()

    # answer!
    answer_query(VDB, LLM, query, thinking)
