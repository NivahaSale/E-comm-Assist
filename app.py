import streamlit as st
from agent import run_agent

st.set_page_config(page_title="🛒 E-Comm Assist", page_icon="🤖", layout="wide")
st.title("🛍️ E-Comm Assist - Smart Order & Returns Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_query = st.chat_input("Ask about your order, returns, or products...")
if user_query:
    with st.spinner("Thinking..."):
        response = run_agent(user_query)
        st.session_state.chat_history.append(("You", user_query))
        st.session_state.chat_history.append(("E-Comm Assist", response))

# Display conversation
for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)
