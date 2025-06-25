# app/run_localGPT.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import get_embeddings
from langchain.vectorstores import FAISS
from localgpt.load_models import load_model
from localgpt.constants import PERSIST_DIRECTORY

def ask_question(query: str) -> str:
    embeddings = get_embeddings()
    vectordb = FAISS.load_local(PERSIST_DIRECTORY, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever()

    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    full_prompt = f"{context}\n\n{query}"

    llm = load_model()
    response = llm.invoke(full_prompt)
    return response

# For CLI use (optional)
def main():
    print("💬 Ask questions about your CSV data (type 'exit' to quit):")
    while True:
        query = input("\n🧠 You: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = ask_question(query)
        print(f"🤖 Bot: {answer}")

if __name__ == "__main__":
    main()
