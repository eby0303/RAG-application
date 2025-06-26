# app/utils.py

from langchain_community.llms import LlamaCpp
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from localgpt.utils import load_config

def get_embeddings():
    model_path = "models/local_e5_model"
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu"}
    )


def load_model():
    config = load_config()
    model_path = config["model_name"]

    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=2048,
        temperature=0.2,
        top_p=0.95,
        repeat_penalty=1.1,
        n_threads=3,  # Set lower if you're on a low-end CPU
        verbose=True,
        streaming=False,
        f16_kv=False,
    )
    return llm
