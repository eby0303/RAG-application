# app/run_localgpt.py

from langchain_community.vectorstores import FAISS
from app.utils import load_model, get_embeddings
from localgpt.constants import PERSIST_DIRECTORY

def ask_question(query: str):
    embeddings = get_embeddings()
    vectordb = FAISS.load_local(PERSIST_DIRECTORY, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever()

    llm = load_model()
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    full_prompt = f"""
    You are a telecom data analyst.
    Use the following context to answer the user query:

    Context:
    {context}

    User query: {query}

    Available metrics:
    - prov_sub = provisioned subscribers
    - act_sub = active subscribers
    - att_sub = attached subscribers

    Instructions:
    - Only respond in valid JSON (no commentary or markdown).
    - Ensure the JSON strictly follows the format below.
    - Pick the most relevant metric and chart type based on the user's intent.

    Respond **with ONLY** a valid JSON object in this format — no extra text, explanation, or markdown:
    {{
      "analysis": "...",
      "chart_type": "...",
      "metric": "...",
      "region": "...",
      "date_range": ["YYYY-MM-DD", "YYYY-MM-DD"]
    }}
    """.strip()


    print(f"\n=== Prompt to LLM ===\n{full_prompt}\n")
    response = llm.invoke(full_prompt)
    print(f"\n=== LLM Response ===\n{full_prompt}\n")
    print(response)
    return response, docs