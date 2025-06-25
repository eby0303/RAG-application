# app/utils.py

from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.llms import LlamaCpp

def get_embeddings():
    return LlamaCppEmbeddings(
        model_path="models/e5-small-v2.Q4_K_M.gguf",
        n_ctx=2048,
        n_threads=4,
        f16_kv=False
    )

def load_llm():
    return LlamaCpp(
        model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        n_ctx=2048,
        temperature=0.2,
        verbose=True,
        streaming=False
    )
