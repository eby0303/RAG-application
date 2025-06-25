# localgpt/load_models.py

from langchain_community.llms import LlamaCpp
from localgpt.utils import load_config

def load_model():
    config = load_config()
    model_path = config["model_name"]

    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=2048,
        temperature=0.2,
        top_p=0.95,
        repeat_penalty=1.1,
        n_threads=4,  # Set lower if you're on a low-end CPU
        verbose=True,
        streaming=False
    )
    return llm
