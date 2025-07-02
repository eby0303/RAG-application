# localgpt/utils.py

import os
import csv
import yaml
from langchain_core.documents import Document
from localgpt.constants import CSV_EXTENSIONS

def load_document_batch(filepaths: list[str]) -> tuple[list[Document], list[str]]:
    all_docs = []
    failed_files = []
    for path in filepaths:
        ext = os.path.splitext(path)[1].lower()
        if ext in CSV_EXTENSIONS:
            try:
                with open(path, "r", encoding="utf-8") as csvfile:
                    reader = csv.reader(csvfile)
                    next(reader, None)  # Skip header
                    for idx, row in enumerate(reader):
                        content = ", ".join(row)
                        metadata = {"source": path, "row": idx}
                        all_docs.append(Document(page_content=content, metadata=metadata))
            except Exception as e:
                print(f" Failed to read CSV: {path} — {e}")
                failed_files.append(path)
        else:
            failed_files.append(path)
    return all_docs, failed_files

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
