# app/run_localGPT.py
from langchain_community.vectorstores import FAISS
from app.utils import load_model, get_embeddings
from localgpt.constants import PERSIST_DIRECTORY
import json
import re

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

def ask_question(query: str):
    embeddings = get_embeddings()
    vectordb = FAISS.load_local(PERSIST_DIRECTORY, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever()
    
    llm = load_model()

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
        "chart_type": "line" or "bar" or "area",
        "x_axis": "x-axis label (e.g., date, region, etc)",
        "y_axis": "y-axis label (metric name)",
        "series": {{
            "label_1": [x1, x2, x3, ...],  # could be dates, categories, metrics, etc. defines the x-axis values for each label/metric
            "label_2": [x1, x2, x3, ...]   # Keys are labels that will appear in the chart legend
        }},
        "values": {{
            "label_1": [y1, y2, y3, ...], # 'values' provides the corresponding y-axis values for each label
            "label_2": [y1, y2, y3, ...]  # Must match the length of the corresponding entry in 'series'
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
    """.strip()


    print(f"\n=== Prompt to LLM ===\n{prompt}\n")
    response = llm.invoke(prompt)
    print(f"\n=== LLM Response ===\n{response}\n")
    json_data = extract_json(response)
    if json_data:
        return json_data, docs
    else:
        print("Could not extract valid JSON")
        return None, docs
