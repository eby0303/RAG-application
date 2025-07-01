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
        parsed, retrieved_docs = ask_question(user_prompt)

    if parsed is None:
        st.error("❌ Could not extract valid JSON from LLM response.")
        st.stop()

    st.subheader("📈 Analysis")
    st.write(parsed["analysis"])

    # Show chart only if flagged
    if parsed.get("show_chart", False):
        # Read source data
        df = pd.concat([
            pd.read_csv(f"data/source_documents/{f}")
            for f in os.listdir("data/source_documents") if f.endswith(".csv")
        ])
        df["date"] = pd.to_datetime(df["date"])

        region = parsed.get("region")
        metrics = parsed.get("metrics", [])
        chart_type = parsed.get("chart_type")
        start_date, end_date = pd.to_datetime(parsed["date_range"][0]), pd.to_datetime(parsed["date_range"][1])

        chart_df = df[(df["circle"] == region) & (df["date"].between(start_date, end_date))]
        chart_df = chart_df.sort_values("date")

        st.subheader("📉 Suggested Chart")
        if metrics:
            chart_data = chart_df[["date"] + metrics].set_index("date")

            if chart_type == "line":
                st.line_chart(chart_data)
            elif chart_type == "bar":
                st.bar_chart(chart_data)
            elif chart_type == "area":
                st.area_chart(chart_data)
            else:
                st.warning("Unsupported chart type returned by the model.")
        else:
            st.warning("⚠️ No metrics provided for chart.")
        
        # KPI Summary
        if "kpi_summary" not in parsed:
            kpi_summary = {}
            for metric in metrics:
                kpi_summary[metric] = {
                    "min": float(chart_df[metric].min()),
                    "max": float(chart_df[metric].max()),
                    "mean": float(chart_df[metric].mean())
                }
        else:
            kpi_summary = parsed["kpi_summary"]

        st.subheader("📊 KPI Summary")
        st.json(kpi_summary)    

    else:
        st.info("ℹ️ No chart was requested by the LLM for this query.")
        

    # 💡 Insights section
    if "insights" in parsed and isinstance(parsed["insights"], list):
        st.subheader("💡 Insights")
        for idx, insight in enumerate(parsed["insights"], 1):
            st.markdown(f"- {insight}")

    with st.expander("📄 Retrieved Context"):
        for i, doc in enumerate(retrieved_docs):
            st.markdown(f"**Chunk {i+1}:**\n{doc.page_content}\n")
