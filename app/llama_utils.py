# app/gemini_utils.py

import os
import json
import re
import requests
from app.utils import get_embeddings
from langchain_community.vectorstores import FAISS
from localgpt.constants import PERSIST_DIRECTORY

from dotenv import load_dotenv
load_dotenv()


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost",  # Optional
    "X-Title": "Telecom-RAG-App"         # Optional
}
LLAMA_MODEL = "meta-llama/llama-3.1-8b-instruct:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def extract_json(text: str):
    try:
        json_candidate = re.search(r'\{.*\}', text.strip(), re.DOTALL)
        if json_candidate:
            cleaned = json_candidate.group().strip("`").strip()
            return json.loads(cleaned)
    except Exception as e:
        print(f"❌ Error extracting JSON: {e}")
    return None

def ask_question_llama(query: str):
    embeddings = get_embeddings()
    vectordb = FAISS.load_local(PERSIST_DIRECTORY, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})
    docs = retriever.get_relevant_documents(query)

    context = "\n".join(doc.page_content.strip() for doc in docs if doc.page_content.strip())

    prompt = f"""
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

REMEMBER TO PROVIDE ONLY A COMPLETE JSON AND NO EXPLANATION OR COMMENTS WITH IT
""".strip()

    print(f"\n=== Prompt to LLM ===\n{prompt}\n")

    payload = {
        "model": LLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=HEADERS, data=json.dumps(payload))
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"]
        print(f"\n=== LLaMA Response ===\n{reply}\n")
        json_data = extract_json(reply)
        return json_data, docs
    except requests.exceptions.RequestException as e:
        print(f"❌ Error from OpenRouter API: {e} - {response.text}")
        return None, docs