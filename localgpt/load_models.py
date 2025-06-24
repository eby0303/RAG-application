from langchain_community.llms import LlamaCpp
from localgpt.utils import load_config

config = load_config()
model_path = config["model_name"]  # This should be the path to your GGUF model

def load_model():
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=2048,
        temperature=0.1,
        top_p=0.95,
        repeat_penalty=1.1,
        n_threads=4,  # You can reduce this if you face CPU overload
        verbose=False
    )
    return llm
