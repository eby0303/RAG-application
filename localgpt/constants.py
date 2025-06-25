# localgpt/constants.py

# Only allow CSV files
CSV_EXTENSIONS = [".csv"]

# Text splitting config
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# File paths
SOURCE_DIRECTORY = "data/source_documents"  # Where CSV files are placed
PERSIST_DIRECTORY = "data/indexes"          # Where FAISS index is saved
