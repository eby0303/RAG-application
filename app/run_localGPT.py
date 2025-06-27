# app/run_localgpt.py

from langchain.vectorstores import FAISS
from app.utils import load_model, get_embeddings
from localgpt.constants import PERSIST_DIRECTORY

# Cache to avoid reloading every time
llm = None
retriever = None
embeddings = None

def ask_question(query: str) -> str:
    global llm, retriever, embeddings

    # Load embeddings once
    if embeddings is None:
        embeddings = get_embeddings()

    # Load FAISS retriever once
    if retriever is None:
        vectordb = FAISS.load_local(PERSIST_DIRECTORY, embeddings, allow_dangerous_deserialization=True)
        retriever = vectordb.as_retriever()

    # Load LLM once
    if llm is None:
        llm = load_model()

    # Retrieve and build prompt
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    full_prompt = f"{context}\n\n{query}"

    print(f"\n=== Prompt to LLM ===\n{full_prompt}\n")
    response = llm.invoke(full_prompt)
    return response
