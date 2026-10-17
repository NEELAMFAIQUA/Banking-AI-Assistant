"""Streamlit interface for the educational banking information assistant."""

from __future__ import annotations

import streamlit as st
from openai import OpenAIError

from banking_assistant.generation import generate_answer, llm_is_configured
from banking_assistant.retrieval import load_knowledge_base, retrieve

st.set_page_config(page_title="Banking Information Assistant", page_icon="🏦")
st.title("Banking Information Assistant")
st.caption(
    "A learning tool for general banking topics, grounded in a small reference library."
)
st.warning(
    "Educational information only—not personalized financial advice. "
    "This app cannot access accounts or perform transactions. Do not enter "
    "account numbers, passwords, PINs, or verification codes."
)

if llm_is_configured():
    st.caption("Answer mode: retrieved references with optional AI generation.")
else:
    st.caption("Answer mode: local retrieval demo (no API key configured).")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a general banking question")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    matches = retrieve(question, load_knowledge_base())
    with st.chat_message("assistant"):
        if not matches:
            answer = (
                "I couldn't find a relevant topic in the reference library. "
                "For account-specific help, contact your bank through an official channel."
            )
        elif llm_is_configured():
            try:
                answer = generate_answer(question, matches)
            except OpenAIError as error:
                st.error(f"AI answer generation failed: {error}")
                answer = (
                    "I couldn't generate an AI answer. Here are the relevant "
                    "reference excerpts instead."
                )
        else:
            answer = (
                "AI-generated answers are disabled in local demo mode. "
                "Here are the closest reference excerpts:"
            )

        st.markdown(answer)
        if matches:
            with st.expander("Reference excerpts"):
                for article in matches:
                    st.markdown(f"**{article['title']}**")
                    st.write(article["answer"])
                    st.caption(f"Source: {article['source']}")
        st.session_state.messages.append({"role": "assistant", "content": answer})
