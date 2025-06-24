from langchain_community.embeddings import LlamaCppEmbeddings

def get_embeddings():
    return LlamaCppEmbeddings(
        model_path="models/e5-small-v2.Q4_K_M.gguf",
    )# localgpt/utils.py

import os
import csv
from langchain_core.documents import Document
from localgpt.constants import CSV_EXTENSIONS
import yaml

def load_csv_as_documents(file_path: str) -> list[Document]:
    documents = []
    try:
        with open(file_path, "r", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)
            for idx, row in enumerate(reader):
                content = ", ".join(row)
                metadata = {"source": file_path, "row": idx}
                documents.append(Document(page_content=content, metadata=metadata))
    except Exception as e:
        print(f"❌ Failed to read CSV: {file_path} — {e}")
    return documents

def load_document_batch(filepaths: list[str]) -> tuple[list[Document], list[str]]:
    all_docs = []
    failed_files = []
    for path in filepaths:
        ext = os.path.splitext(path)[1].lower()
        if ext in CSV_EXTENSIONS:
            docs = load_csv_as_documents(path)
            all_docs.extend(docs)
        else:
            failed_files.append(path)
    return all_docs, failed_files

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
