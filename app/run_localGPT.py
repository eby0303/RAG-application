# app/run_localGPT.py

from langchain_community.vectorstores import FAISS
from app.utils import load_model, get_embeddings
from localgpt.constants import PERSIST_DIRECTORY
import json
import re

def extract_region(query: str) -> str:
    # You can expand this map as needed
    region_map = {
        "maharashtra": "MH",
        "madhya pradesh": "MP",
        "mumbai": "MU",
        "north east": "NE"
    }

    query_lower = query.lower()
    for name, code in region_map.items():
        if name in query_lower or code.lower() in query_lower:
            return code
    return None  # fallback if not found

def extract_json(text: str):
    try:
        json_candidate = re.search(r'\{.*\}', text.strip(), re.DOTALL)
        if json_candidate:
            cleaned = json_candidate.group().strip("`").strip()
            return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error during JSON extraction: {e}")
    return None


def ask_question(query: str):
    embeddings = get_embeddings()
    vectordb = FAISS.load_local(PERSIST_DIRECTORY, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever()

    region_code = extract_region(query)
    llm = load_model()

    docs = retriever.get_relevant_documents(query)

    if region_code:
        docs = [doc for doc in docs if region_code.lower() in doc.page_content.lower()]

    # context = "\n\n".join(doc.page_content for doc in docs)
    context = """
On June 21, 2025 (i.e., 2025-06-21), in Maharashtra (circle code: MH), there were 112630 provisioned subscribers, 98593 active subscribers, and 97120 attached subscribers, with an attach rate of 98.5% and an active rate of 87.5%.
On June 22, 2025 (i.e., 2025-06-22), in Maharashtra (circle code: MH), there were 108366 provisioned subscribers, 92973 active subscribers, and 89440 attached subscribers, with an attach rate of 96.2% and an active rate of 85.8%.
On June 23, 2025 (i.e., 2025-06-23), in Maharashtra (circle code: MH), there were 112532 provisioned subscribers, 97878 active subscribers, and 95009 attached subscribers, with an attach rate of 97.1% and an active rate of 87.0%.
On June 24, 2025 (i.e., 2025-06-24), in Maharashtra (circle code: MH), there were 116663 provisioned subscribers, 102960 active subscribers, and 102057 attached subscribers, with an attach rate of 99.1% and an active rate of 88.3%.
""".strip()

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
      "region": "{region_code}",
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
        print("❌ Could not extract valid JSON")
        return None, docs
