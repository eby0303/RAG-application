import os
import pandas as pd
import shutil
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from utils import get_embeddings
from localgpt.constants import CHUNK_SIZE, CHUNK_OVERLAP, SOURCE_DIRECTORY, PERSIST_DIRECTORY

# Mapping for circle code to name
CIRCLE_CODE_TO_NAME = {
    "AP": "Andhra Pradesh", "AS": "Assam", "BH": "Bihar", "CH": "Chennai", "DL": "Delhi",
    "GU": "Gujarat", "HP": "Himachal Pradesh", "HR": "Haryana", "JK": "Jammu and Kashmir", "KA": "Karnataka",
    "KL": "Kerala", "KO": "Kolkata", "MH": "Maharashtra", "MP": "Madhya Pradesh", "MU": "Mumbai",
    "NE": "North East", "OR": "Odisha", "PB": "Punjab", "RJ": "Rajasthan", "TN": "Tamil Nadu",
    "UE": "Uttar Pradesh (East)", "UW": "Uttar Pradesh (West)"
}

def load_csv_documents(source_dir):
    documents = []

    for file_name in os.listdir(source_dir):
        if file_name.endswith(".csv"):
            df = pd.read_csv(os.path.join(source_dir, file_name))

            for _, row in df.iterrows():
                circle_code = row["circle"]
                circle_name = CIRCLE_CODE_TO_NAME.get(circle_code, circle_code)
                prov = row["prov_sub"]
                act = row["act_sub"]
                att = row["att_sub"]
                iso_date = row["date"]

                dt = pd.to_datetime(iso_date)
                readable_date = dt.strftime("%B %d, %Y")

                sentence = (
                    f"On {readable_date} (i.e., {iso_date}), in {circle_name} (circle code: {circle_code}), "
                    f"there were {prov} prov_sub, {act} act_sub, and {att} att_sub."
                )

                #  One row = one Document
                documents.append(Document(page_content=sentence, metadata={
                    "circle_code": circle_code,
                    "circle_name": circle_name,
                    "date": iso_date,
                    "source": file_name
                }))

    return documents


def update_faiss():
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
        print(f"🧹 Old FAISS index at {PERSIST_DIRECTORY} removed")

    print(f"\U0001F4C2 Loading CSV files from {SOURCE_DIRECTORY}")
    documents = load_csv_documents(SOURCE_DIRECTORY)
    print(f"✅ Loaded {len(documents)} documents")

    text_chunks = documents 
    print(f"📄 Split into {len(text_chunks)} chunks")

    embeddings = get_embeddings()
    vectordb = FAISS.from_documents(text_chunks, embedding=embeddings)
    vectordb.save_local(PERSIST_DIRECTORY)
    print(f"✅ FAISS DB saved at {PERSIST_DIRECTORY}")

def main():
    update_faiss()
