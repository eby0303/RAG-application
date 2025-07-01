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

def ask_question_llama(query: str, k: int):
    embeddings = get_embeddings()
    vectordb = FAISS.load_local(PERSIST_DIRECTORY, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
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

    If asked to compare multiple regions, include multiple region names.
    If analyzing a single region, do not repeat the same region name multiple times in 'series' and 'values'.
    Ensure that all keys in 'series' and 'values' are unique and match valid region codes only once.


    Respond ONLY in this JSON format:

    {{
    "analysis": "Your overall comparative analysis here.",
    "show_chart": true,
    "charts": [
        {{
        "title": "Your Chart Title",
        "chart_type": "line" or "bar" or "area",
        "x_axis": "date",
        "y_axis": "metric_name",
        "series": {{
            "region_name_1": ["YYYY-MM-DD", ...], #series should be a dictionary of region → list of dates.
            "region_name_2": ["YYYY-MM-DD", ...]
        }},
        "values": {{
            "region_name_1": [val1, val2, ...], #values should be list of numbers (for that one metric only)
            "region_name_2": [val1, val2, ...]
        }}
        }}
    ],
    "kpi_summary": {{
        "region_name_1": {{
        "prov_sub": {{"min": 12345, "max": 23456, "mean": 19000.5}},
        "act_sub": {{"min": 10000, "max": 21000, "mean": 17000.0}},
        "att_sub": {{"min": 12345, "max": 23456, "mean": 19000.5}}
        }},
        "region_name_2": {{
        "prov_sub": {{...}},
        "act_sub": {{...}},
        "att_sub": {{...}}
        }}
    }},
    "insights": {{
        "individual": {{
        "region_name_1": ["insight1", "insight2"], #provide insights for each region only no comparison here
        "region_name_2": ["insight1", "insight2"]
        }},
        "collective": [
        "comparison insight 1", #provide insights for all regions here
        "comparison insight 2"
        ]
    }}
    }}

    REMEMBER: Return only valid JSON. No COMMENTS or EXPLANATIONS outside or inside the JSON. Never include trailing commas.
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
