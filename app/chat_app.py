# app/chat_app.py
import sys, os
import json
import re
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.run_localGPT import ask_question
from app import ingest

st.title("📊 Telecom Analytics Assistant")

if st.button("🔄 Update FAISS DB"):
    with st.spinner("Re-indexing... this may take a few moments..."):
        ingest.update_faiss()
    st.success("FAISS DB updated successfully!")

user_prompt = st.text_input("Ask your question:", placeholder="e.g., Show trends in MH for June 2025")

if user_prompt:
    with st.spinner("🔍 Thinking..."):
        response, retrieved_docs = ask_question(user_prompt)

    # 🛠 Robust JSON extraction
    try:
        # Try direct JSON load
        parsed = json.loads(response)
    except json.JSONDecodeError:
        # Try extracting JSON via regex
        match = re.search(r"\{\s*\"analysis\".*?\}", response, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except Exception as e:
                st.error(f"❌ Still couldn't parse cleaned JSON.\nError: {e}\n\nRaw response:\n{response}")
                st.stop()
        else:
            st.error(f"❌ Could not extract valid JSON from model output.\n\nRaw response:\n{response}")
            st.stop()

    # ✅ Valid JSON at this point
    st.subheader("📈 Analysis")
    st.write(parsed["analysis"])

    # Read source data
    df = pd.concat([
        pd.read_csv(f"data/source_documents/{f}")
        for f in os.listdir("data/source_documents") if f.endswith(".csv")
    ])
    df["date"] = pd.to_datetime(df["date"])

    region = parsed["region"]
    metric = parsed["metric"]
    chart_type = parsed["chart_type"]
    start_date, end_date = pd.to_datetime(parsed["date_range"][0]), pd.to_datetime(parsed["date_range"][1])

    chart_df = df[(df["circle"] == region) & (df["date"].between(start_date, end_date))]
    chart_df = chart_df.sort_values("date")

    st.subheader("📉 Suggested Chart")
    chart_data = chart_df[["date", metric]].set_index("date")

    if chart_type == "line":
        st.line_chart(chart_data)
    elif chart_type == "bar":
        st.bar_chart(chart_data)
    elif chart_type == "area":
        st.area_chart(chart_data)
    else:
        st.warning("Unsupported chart type returned by the model.")

    with st.expander("📄 Retrieved Context"):
        for i, doc in enumerate(retrieved_docs):
            st.markdown(f"**Chunk {i+1}:**\n{doc.page_content}\n")
