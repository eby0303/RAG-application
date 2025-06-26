import os
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from app.utils import get_embeddings
from localgpt.constants import CHUNK_SIZE, CHUNK_OVERLAP, SOURCE_DIRECTORY, PERSIST_DIRECTORY

def load_csv_documents(source_dir):
    documents = []
    for file_name in os.listdir(source_dir):
        if file_name.endswith(".csv"):
            df = pd.read_csv(os.path.join(source_dir, file_name))
            for _, row in df.iterrows():
                content = " | ".join(str(cell) for cell in row)
                documents.append(Document(page_content=content, metadata={"source": file_name}))
    return documents

def main():
    print(f"📂 Loading CSV files from {SOURCE_DIRECTORY}")
    documents = load_csv_documents(SOURCE_DIRECTORY)
    print(f"✅ Loaded {len(documents)} rows")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    text_chunks = splitter.split_documents(documents)
    print(f"📄 Split into {len(text_chunks)} chunks")

    embeddings = get_embeddings()
    vectordb = FAISS.from_documents(text_chunks, embedding=embeddings)
    vectordb.save_local(PERSIST_DIRECTORY)
    print(f"✅ FAISS DB saved at {PERSIST_DIRECTORY}")

if __name__ == "__main__":
    main()
