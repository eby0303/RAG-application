# app/run_localGPT.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain.chains import RetrievalQA
from utils import get_embeddings
from langchain.vectorstores import Chroma
from localgpt.load_models import load_model
from localgpt.constants import PERSIST_DIRECTORY

def ask_question(query: str) -> str:
    embeddings = get_embeddings()
    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    retriever = vectordb.as_retriever()
    llm = load_model()

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
    result = qa.run(query)
    return result

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
