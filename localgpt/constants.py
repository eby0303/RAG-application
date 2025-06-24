# localgpt/constants.py

CSV_EXTENSIONS = [".csv"]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

CHROMA_SETTINGS = {
    "persist_directory": "data/indexes",
    "anonymized_telemetry": False,
}

SOURCE_DIRECTORY = "data/SOURCE_DOCUMENTS"
PERSIST_DIRECTORY = "data/indexes" 