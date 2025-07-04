# app/chat_app.py

import sys, os
import json
import re
import pandas as pd
import streamlit as st
from dateutil.parser import parse as date_parse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.run_localGPT import ask_question
from app.llama_utils import ask_question_llama
from app import ingest

def is_date_like(val):
    try:
        date_parse(str(val))
        return True
    except Exception:
        return False

st.set_page_config(page_title="Telecom Analytics", layout="wide")
st.title("Telecom Analytics Assistant")

# --- Sidebar Filters ---
st.sidebar.header(" Settings")

# Retrieval chunk count
k_val = st.sidebar.slider("Chunks to retrieve (k)", min_value=1, max_value=20, value=10)

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
        st.markdown(parsed["analysis_md"], unsafe_allow_html=True)

    
    # Show chart if requested by LLM
    if parsed.get("show_chart", False):
        charts = parsed.get("charts", [])
        for chart in charts:
            st.subheader(chart.get("title", "Chart"))

            chart_type = chart.get("chart_type", "line")
            x_axis = chart.get("x_axis", "x")
            y_axis = chart.get("y_axis", "y")

            all_dfs = []
            auto_detect_dates = False

            series = chart.get("series", {})
            values = chart.get("values", {})

            for label, x_vals in series.items():
                y_vals = values.get(label, [])

                if not x_vals:
                    continue

                # Align lengths
                if len(y_vals) < len(x_vals):
                    y_vals += [0] * (len(x_vals) - len(y_vals))
                elif len(y_vals) > len(x_vals):
                    y_vals = y_vals[:len(x_vals)]

                df = pd.DataFrame({x_axis: x_vals, label: y_vals})

                # if x-axis values look like dates
                if is_date_like(x_vals[0]):
                    auto_detect_dates = True
                    df[x_axis] = pd.to_datetime(df[x_axis], errors="coerce")

                df.dropna(subset=[x_axis], inplace=True)
                df.set_index(x_axis, inplace=True)
                all_dfs.append(df)

            if all_dfs:
                chart_df = pd.concat(all_dfs, axis=1)

                # Plot based on type
                if chart_type == "line":
                    st.line_chart(chart_df)
                elif chart_type == "bar":
                    st.bar_chart(chart_df)
                elif chart_type == "area":
                    st.area_chart(chart_df)
                elif chart_type == "scatter":
                    st.scatter_chart(chart_df)
                else:
                    st.warning(f"Unsupported chart type: {chart_type}")
            else:
                st.info("No valid series data to render chart.")


    # Markdown-based insights
    if "insights_md" in parsed:
        st.markdown(parsed["insights_md"], unsafe_allow_html=True)

    with st.expander(" Retrieved Context"):
        for i, doc in enumerate(retrieved_docs):
            st.markdown(f"**Chunk {i+1}:**\n{doc.page_content}\n")
            
            
            
            
