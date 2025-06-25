from localgpt.constants import CHUNK_SIZE, CHUNK_OVERLAP
from app.utils import get_embeddings
from langchain_community.vectorstores import Chroma

PERSIST_DIRECTORY = "db"

def test_query(query: str):
    # Load the embeddings and vector database
    embeddings = get_embeddings()
    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

    # Embed the query
    print(f"🔍 Query: {query}")
    results = vectordb.similarity_search(query, k=3)

    # Display retrieved chunks
    print("\n📎 Top matching chunks:\n")
    for i, doc in enumerate(results, 1):
        print(f"🔹 Chunk {i}:")
        print(doc.page_content)
        print("-" * 40)

if __name__ == "__main__":
    test_query("What is the salary of Alice?")
