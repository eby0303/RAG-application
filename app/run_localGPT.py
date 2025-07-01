# app/run_localGPT.py

from langchain_community.vectorstores import FAISS
from app.utils import load_model, get_embeddings
from localgpt.constants import PERSIST_DIRECTORY
import json
import re

def extract_json(text: str):
    try:
        json_candidate = re.search(r'\{.*\}', text.strip(), re.DOTALL)
        if json_candidate:
            cleaned = json_candidate.group().strip("`").strip()
            return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
    except Exception as e:
        print(f" Unexpected error during JSON extraction: {e}")
    return None


def ask_question(query: str):
    embeddings = get_embeddings()
    vectordb = FAISS.load_local(PERSIST_DIRECTORY, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever()
    
    llm = load_model()

    docs = retriever.get_relevant_documents(query)
    context = "\n".join(doc.page_content.strip() for doc in docs if doc.page_content.strip())

    full_prompt = f"""
    You are a telecom analyst. Based on the data below, answer the user's question by returning only valid JSON.

    Context:
    {context}

    User question:
    {query}

    Available metrics:
    - prov_sub (provisioned subscribers)
    - act_sub (active subscribers)
    - att_sub (attached subscribers)

    Respond ONLY in this JSON format:

    {{
      "analysis": "your analytical summary here",
      "show_chart": true,
      "chart_type": "line" or "bar" or "area",
      "metrics": ["prov_sub", "act_sub", "att_sub"],
      "region": "circle code",
      "date_range": ["YYYY-MM-DD", "YYYY-MM-DD"],
      "kpi_summary": {{
        "prov_sub": {{"min": 12345, "max": 23456, "mean": 19000.5}},
        "act_sub": {{"min": 10000, "max": 21000, "mean": 17000.0}},
        "att_sub": {{"min": 12345, "max": 23456, "mean": 19000.5}}
      }},
      "insights": ["Insight"]
    }}
    
    REMEMBER TO PROVIDE A COMPLETE JSON
    """.strip()

    print(f"\n=== Prompt to LLM ===\n{full_prompt}\n")
    response = llm.invoke(full_prompt)
    print(f"\n=== LLM Response ===\n{response}\n")
    json_data = extract_json(response)
    if json_data:
        return json_data, docs
    else:
        print("Could not extract valid JSON")
        return None, docs
