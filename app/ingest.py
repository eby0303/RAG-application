import os
import json
import shutil
import pandas as pd
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
import sys

from utils import get_embeddings
from localgpt.constants import SOURCE_DIRECTORY, PERSIST_DIRECTORY

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SOURCE_DIRECTORY = "data/SOURCE_DOCUMENTS1"

# Circle code mapping
CIRCLE_CODE_TO_NAME = {
    "AP": "Andhra Pradesh", "AS": "Assam", "BH": "Bihar", "CH": "Chennai", "DL": "Delhi",
    "GU": "Gujarat", "HP": "Himachal Pradesh", "HR": "Haryana", "JK": "Jammu and Kashmir", "KA": "Karnataka",
    "KL": "Kerala", "KO": "Kolkata", "MH": "Maharashtra", "MP": "Madhya Pradesh", "MU": "Mumbai",
    "NE": "North East", "OR": "Odisha", "PB": "Punjab", "RJ": "Rajasthan", "TN": "Tamil Nadu",
    "UE": "Uttar Pradesh (East)", "UW": "Uttar Pradesh (West)"
}

def get_common_metadata(source_dir):
    metadata_map = {}

    for file in os.listdir(source_dir):
        if file.endswith("_metadata.json"):
            try:
                key = file.replace("_metadata.json", "").lower()
                with open(os.path.join(source_dir, file), "r") as f:
                    metadata_map[key] = json.load(f)
            except Exception as e:
                print(f"Error reading metadata {file}: {e}")
    return metadata_map

def infer_metadata_for_file(file_name, metadata_map):
    file_key = file_name.lower().replace(".csv", "")
    for meta_key in metadata_map.keys():
        if meta_key in file_key:
            return metadata_map[meta_key]
    return {}  

def load_csv_documents(source_dir=SOURCE_DIRECTORY):
    documents = []
    metadata_map = get_common_metadata(source_dir)

    for file_name in os.listdir(source_dir):
        if not file_name.endswith(".csv"):
            continue

        csv_path = os.path.join(source_dir, file_name)
        df = pd.read_csv(csv_path)

        # find matching metadata file
        field_metadata = infer_metadata_for_file(file_name, metadata_map)

        for _, row in df.iterrows():
            circle_code = row.get("circle")
            iso_date = row.get("date")
            if not circle_code or not iso_date:
                continue

            circle_name = CIRCLE_CODE_TO_NAME.get(str(circle_code), str(circle_code))
            try:
                readable_date = pd.to_datetime(iso_date).strftime("%B %d, %Y")
            except Exception:
                readable_date = iso_date 

            stats = []
            field_descriptions = {}

            for field in row.index:
                if field in ["circle", "date"]:
                    continue
                value = row[field]
                description = field_metadata.get(field, field)
                stats.append(f"{value} {field}")
                field_descriptions[field] = description

            stat_str = ", ".join(stats)
            sentence = (
                f"On {readable_date} (i.e., {iso_date}), in {circle_name} (circle code: {circle_code}), "
                f"there were {stat_str}."
            )

            doc_metadata = {
                "source": file_name,
                "fields": field_descriptions
            }

            documents.append(Document(page_content=sentence, metadata=doc_metadata))

    return documents

def update_faiss():
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
        print(f"Old FAISS index at {PERSIST_DIRECTORY} removed")

    print(f"Loading CSV files from {SOURCE_DIRECTORY}")
    documents = load_csv_documents(SOURCE_DIRECTORY)
    print(f"Loaded {len(documents)} documents")

    embeddings = get_embeddings()
    vectordb = FAISS.from_documents(documents, embedding=embeddings)
    vectordb.save_local(PERSIST_DIRECTORY)
    print(f"FAISS DB saved at {PERSIST_DIRECTORY}")

def main():
    update_faiss()

if __name__ == "__main__":
    main()
