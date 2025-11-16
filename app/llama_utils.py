# app/gemini_utils.py

import os
import json
import re
import requests
import time
from app.utils import get_embeddings
from langchain_community.vectorstores import FAISS

# Get the project root directory (parent of app directory)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PERSIST_DIRECTORY = os.path.join(PROJECT_ROOT, "data", "indexes")

from dotenv import load_dotenv
load_dotenv()


# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  
OPENROUTER_API_KEY = "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"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",   
    "HTTP-Referer": "http://localhost",  
    "X-Title": "Telecom-RAG-App"   
}
LLAMA_MODEL = "mistralai/mixtral-8x7b-instruct"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def extract_json(text: str):
    try:
        json_candidate = re.search(r'\{.*\}', text.strip(), re.DOTALL)
        if not json_candidate:
            print("No JSON-like content found.")
            return None
        cleaned = json_candidate.group().strip("` \n")
        cleaned = re.sub(r'(?<!\\)\\(?![\\/"bfnrtu])', r'\\\\', cleaned)
        return json.loads(cleaned)
    except Exception as e:
        print(f"Error extracting JSON: {e}")
    return None

def format_context_with_metadata(docs):
    formatted_chunks = []
    for doc in docs:
        meta = doc.metadata
        meta_str = ", ".join(f"{k}: {v}" for k, v in meta.items())
        chunk = f"[Metadata: {meta_str}]\n{doc.page_content.strip()}"
        formatted_chunks.append(chunk)
    return "\n\n".join(formatted_chunks)

def ask_question_llama(query: str, k: int):
    embeddings = get_embeddings()
    vectordb = FAISS.load_local(PERSIST_DIRECTORY, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    
    context = format_context_with_metadata(docs)

    prompt = f"""
    You are a telecom data analyst. Based on the data below, answer the user's question by returning only valid JSON. Ensure your analysis is accurate, concise, and tailored to the user's intent. Use appropriate metrics, comparisons, summaries, and visual cues where applicable.

    Context:
    {context}

    User question:
    {query}

    Respond ONLY in this JSON format:

    {{
    "analysis_md": "Markdown-formatted analysis section.",
    "show_chart": true,
    "charts": [
        {{
        "title": "Your Chart Title",
        "chart_type": "line" or "scatter" or "area", # whichever is appropriate
        "x_axis": "x-axis label (e.g., date, region, etc)",
        "y_axis": "y-axis label (metric name)",
        "series": {{                      
            "label_1": [x1, x2, x3, ...],  # could be dates, categories, metrics, etc. defines the x-axis values for each label/metric
            "label_2": [x1, x2, x3, ...]   # Keys are labels that will appear in the chart legend
        }},
        "values": {{
            "label_a": [y1, y2, y3, ...], # 'values' provides the corresponding y-axis values for each label
            "label_b": [y1, y2, y3, ...]  # Must match the length of the corresponding entry in 'series'
        }},
        "chart_analysis": "Brief explanation of trends, outliers, or comparisons based on the chart"
        }}
    ],
    "insights_md": "Markdown-formatted insights (both individual and comparative, free-form). Use only if required"
    }}

    For the `analysis_md` and `insights_md` sections:
    - Format your answer using **Markdown**
    - Use **LaTeX** for mathematical expressions if deemed necessary (e.g., quartiles, IQR, mean, standard deviation)
    - Use headings, bullet points, and readable formatting
    - Escape all backslashes in LaTeX using double backslashes (\\\\)
    - Use `$...$` for inline LaTeX
    - Use `$$...$$` for block LaTeX

    Overall note:
    - Never include comments or explanations outside the JSON
    - No trailing commas or invalid JSON
    - ONLY ANSWER WHATS ASKED NOTHING MORE TRY TO UNDERSTAND THE USERS NEED
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
        print(f"Error from OpenRouter API: {e} - {response.text}")
        return None, docs