# my_brain.py

import os
import argparse

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document as LC_Document

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1) Load and split your .txt docs
def build_vectorstore(context_dir, persist_dir="chroma_db"):
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
    # persistence is automatic in chroma 0.4.x+, so you can drop db.persist()
    return db

# 2) Load your LLM once
def load_llm(model_name="meta-llama/Llama-3.2-1b-instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    # **remove** device=0 here
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
    )

# 3) Answer a query
def answer_query(db, llm, query, k=3):
    results = db.similarity_search(query, k=k)
    context = "\n\n".join(r.page_content for r in results)

    prompt = f"""### Instruction:
Use the following context to answer the question.

Context:
{context}

### Question:
{query}

### Response:"""

    out = llm(
        prompt,
        max_new_tokens=200,
        temperature=0.3,
        top_p=0.95,
        top_k=50,
        do_sample=False,
    )
    print(out[0]["generated_text"].strip())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Your question for the LLM")
    args = parser.parse_args()

    # build (or reuse) the vectorstore
    VDB = build_vectorstore(context_dir="context")

    # load the LLM
    LLM = load_llm()

    # ask!
    answer_query(VDB, LLM, args.query)

#python3 my-brain.py "what color is the sky?"