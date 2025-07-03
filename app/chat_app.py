# app/chat_app.py

import sys, os
import json
import re
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.run_localGPT import ask_question
from app.llama_utils import ask_question_llama
from app import ingest

st.set_page_config(page_title="Telecom Analytics", layout="wide")
st.title("Telecom Analytics Assistant")

# --- Sidebar Filters ---
st.sidebar.header(" Settings")

# Retrieval chunk count
k_val = st.sidebar.slider("Chunks to retrieve (k)", min_value=1, max_value=20, value=5)

# LLM selection
llm_option = st.sidebar.radio("Choose LLM:", options=["Llama API", "Local LLM"])

# Update FAISS DB
if st.sidebar.button(" Update FAISS Index"):
    with st.spinner("Re-indexing... Please wait..."):
        ingest.update_faiss()
    st.sidebar.success(" FAISS index updated!")

# --- Main Input ---
with st.form("user_query_form"):
    user_prompt = st.text_input("Ask your question:", placeholder="e.g., Compare MH and KL for June 2025")
    submitted = st.form_submit_button("Submit")

if submitted and user_prompt:
    with st.spinner(" Thinking..."):
        if llm_option == "Local LLM":
            parsed, retrieved_docs = ask_question(user_prompt)
        else:
            parsed, retrieved_docs = ask_question_llama(user_prompt, k=k_val)

    if parsed is None:
        st.error(" Could not extract valid JSON from LLM response.")
        st.stop()

    if "analysis_md" in parsed:
        st.subheader("📊 Analysis")
        st.markdown(parsed["analysis_md"], unsafe_allow_html=True)

    # Show chart if needed
    if parsed.get("show_chart", False):
        charts = parsed.get("charts", [])
        for chart in charts:
            st.subheader(f" {chart.get('title', 'Chart')}")

            chart_type = chart.get("chart_type", "line")
            x_axis = chart.get("x_axis", "date")
            y_axis = chart.get("y_axis", "value")

            region_dataframes = []

            for region, dates in chart["series"].items():
                values = chart["values"].get(region, [])
                
                if not dates:
                    continue

                # If values are shorter than dates, pad with 0s
                if len(values) < len(dates):
                    values += [0] * (len(dates) - len(values))

                # Align values to dates (shortest length to be safe)
                aligned = list(zip(dates, values))
                df = pd.DataFrame(aligned, columns=[x_axis, region])
                df[x_axis] = pd.to_datetime(df[x_axis], errors="coerce")  # handle any "null"
                df.dropna(inplace=True)  # drop any rows with invalid date
                df.set_index(x_axis, inplace=True)
                region_dataframes.append(df)

            if region_dataframes:
                chart_df = pd.concat(region_dataframes, axis=1)
            # Plot
            if chart_type == "line":
                st.line_chart(chart_df)
            elif chart_type == "bar":
                st.bar_chart(chart_df)
            elif chart_type == "area":
                st.area_chart(chart_df)
            else:
                st.warning(" Unsupported chart type.")

    else:
        st.info("ℹ No chart was requested by the LLM for this query.")

    # Markdown-based insights
    if "insights_md" in parsed:
        st.subheader("💡 Insights")
        st.markdown(parsed["insights_md"], unsafe_allow_html=True)

    with st.expander(" Retrieved Context"):
        for i, doc in enumerate(retrieved_docs):
            st.markdown(f"**Chunk {i+1}:**\n{doc.page_content}\n")
