import os
import json
import requests
import streamlit as st

# Config
API_BASE = os.getenv("HNU_API_BASE", "http://localhost:8000")
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))
APP_TITLE = os.getenv("APP_TITLE", "HNU Chatbot")

st.set_page_config(page_title=APP_TITLE, page_icon="🎓", layout="centered")
st.title("🎓 HNU RAG Chat")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    api_base = st.text_input("API base URL", value=API_BASE, help="Your FastAPI server base URL")
    top_k = st.number_input("Top K", min_value=1, max_value=20, value=DEFAULT_TOP_K)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])  # already Markdown-safe content

# Chat input
if prompt := st.chat_input("Frage an die HNU stellen…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend
    try:
        url = f"{api_base.rstrip('/')}/chat"
        payload = {"query": prompt, "top_k": int(top_k)}
        resp = requests.post(url, data=json.dumps(payload), headers={"Content-Type": "application/json"}, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        answer = data.get("answer", "")
        citations = data.get("citations", [])

        # Build assistant message with citations in Markdown
        md = answer or "(Keine Antwort erhalten)"
        if citations:
            md += "\n\n---\nQuellen:\n"
            for i, c in enumerate(citations, 1):
                title = c.get("title") or c.get("url") or "Quelle"
                url = c.get("url") or ""
                if url:
                    md += f"\n[{i}] [{title}]({url})"
                else:
                    md += f"\n[{i}] {title}"

        with st.chat_message("assistant"):
            st.markdown(md)
        st.session_state.messages.append({"role": "assistant", "content": md})

    except requests.RequestException as e:
        err = f"Fehler beim Aufruf der API: {e}"
        with st.chat_message("assistant"):
            st.error(err)
        st.session_state.messages.append({"role": "assistant", "content": err})
