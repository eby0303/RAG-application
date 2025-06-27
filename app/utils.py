# app/utils.py

from langchain_community.llms import LlamaCpp
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from localgpt.utils import load_config
import json

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
        n_threads=4, 
        verbose=True,
        streaming=False,
        f16_kv=False,
    )
    return llm




def load_schema_summary(path="data/metadata_schema.json"):
    with open(path, "r") as f:
        schema = json.load(f)

    field_descriptions = []
    for field, props in schema["fields"].items():
        desc = props["description"]
        field_descriptions.append(f"- {field}: {desc}")

    schema_summary = "Dataset Schema:\n" + "\n".join(field_descriptions)
    return schema_summary