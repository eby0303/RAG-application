# app/chat_app.py
import streamlit as st
from app.run_localGPT import ask_question

st.set_page_config(page_title="Chat with CSV", layout="wide")

st.title("📊 Chat with Your CSV Files")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("Ask a question about your data:")

if st.button("Send") and question:
    answer = ask_question(question)
    st.session_state.chat_history.append(("You", question))
    st.session_state.chat_history.append(("Bot", answer))

for sender, msg in st.session_state.chat_history[::-1]:
    st.markdown(f"**{sender}:** {msg}")
