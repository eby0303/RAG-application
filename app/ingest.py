# app/ingest.py
import os
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from app.utils import get_embeddings
from localgpt.constants import CHUNK_SIZE, CHUNK_OVERLAP

SOURCE_DIRECTORY = "data/SOURCE_DOCUMENTS"
PERSIST_DIRECTORY = "db"

def load_csv_documents(source_dir):
    documents = []
    for file_name in os.listdir(source_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(source_dir, file_name)
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                content = " | ".join(str(cell) for cell in row)
                documents.append(Document(page_content=content, metadata={"source": file_name}))
    return documents

def main():
    print(f"📂 Loading CSV files from {SOURCE_DIRECTORY}")
    documents = load_csv_documents(SOURCE_DIRECTORY)
    print(f"✅ Loaded {len(documents)} rows")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = splitter.split_documents(documents)
    print(f"📄 Split into {len(texts)} chunks")

    embeddings = get_embeddings()
    vectordb = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIRECTORY)
    vectordb.persist()
    print(f"✅ Embeddings stored at {PERSIST_DIRECTORY}")

if __name__ == "__main__":
    main()
