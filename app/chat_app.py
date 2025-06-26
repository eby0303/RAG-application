# app/chat_app.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
from app.run_localGPT import ask_question

st.set_page_config(page_title="Chat with CSV", layout="wide")

st.title("Chat with Your CSV")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("Ask a question about your data:")

if st.button("Send") and question:
    answer = ask_question(question)
    st.session_state.chat_history.append(("You", question))
    st.session_state.chat_history.append(("Bot", answer))

for sender, msg in st.session_state.chat_history[::-1]:
    st.markdown(f"**{sender}:** {msg}")
