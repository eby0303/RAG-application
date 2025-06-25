import os
from langchain_community.vectorstores import Chroma
from app.utils import get_embeddings, load_llm  # assumes your `load_llm()` is defined
from localgpt.constants import CHROMA_SETTINGS

# Set paths
DB_DIR = "db"
QUESTION = "What is the salary of Alice?"

def main():
    print("🔍 Query:", QUESTION)

    # Load embeddings
    embeddings = get_embeddings()
    print("🧠 Loaded embedding model.")

    # Load vector DB
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    print("📚 Vector DB loaded.")

    # Perform similarity search
    results = vectordb.similarity_search(QUESTION, k=3)
    print("\n📎 Top matching chunks:")
    for i, doc in enumerate(results):
        print(f"\n🔹 Chunk {i+1}:\n{doc.page_content}\n{'-'*40}")

    # Join chunks into context
    context = "\n".join([doc.page_content for doc in results])

    # Create prompt
    prompt = f"""Answer the question based on the context below.

Context:
{context}

Question:
{QUESTION}

Answer:"""

    print("\n📝 Final prompt sent to Mistral:\n", prompt)

    # Load local LLM (Mistral)
    llm = load_llm()  # This should return a LlamaCpp pipeline
    print("🤖 Mistral LLM loaded.")

    # Run the LLM
    answer = llm(prompt)
    print("\n💬 Final Answer:\n", answer)

if __name__ == "__main__":
    main()
