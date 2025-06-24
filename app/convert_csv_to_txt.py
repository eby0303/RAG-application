# app/convert_csv_to_txt.py
import os
import pandas as pd
from langchain.docstore.document import Document

def csv_to_documents(csv_path: str) -> list[Document]:
    df = pd.read_csv(csv_path)
    docs = []

    for index, row in df.iterrows():
        content = row.to_string()
        metadata = {"source": csv_path, "row": index}
        docs.append(Document(page_content=content, metadata=metadata))

    return docs

def get_csv_docs_from_directory(source_dir: str) -> list[Document]:
    docs = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(root, file)
                docs.extend(csv_to_documents(csv_path))
    return docs
