"""Optional grounded answer generation; retrieval works without an API key."""

from __future__ import annotations

import os
from typing import Any

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

SYSTEM_PROMPT = """You are a banking information assistant for general education.
Answer only using the supplied reference excerpts. If they do not contain the
answer, say that you cannot determine it from the available information.
Keep answers concise. Do not give personalized financial, legal, or tax advice.
Never request or process passwords, PINs, full account numbers, or other
authentication secrets. You cannot access accounts or perform transactions.
Encourage users to contact their bank through its official channels for
account-specific questions."""


def _setting(name: str, default: str = "") -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        return str(st.secrets.get(name, default)).strip()
    except StreamlitSecretNotFoundError:
        return default


def llm_is_configured() -> bool:
    return bool(_setting("OPENAI_API_KEY"))


def generate_answer(question: str, articles: list[dict[str, Any]]) -> str:
    """Generate an answer grounded in retrieved articles using OpenAI."""
    from openai import OpenAI

    api_key = _setting("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in the environment or Streamlit secrets.")

    excerpts = "\n\n".join(
        f"Source: {article['source']}\n"
        f"Topic: {article['title']}\n"
        f"Reference: {article['answer']}"
        for article in articles
    )
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=_setting("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Reference excerpts:\n{excerpts}\n\nQuestion: {question}",
            },
        ],
        temperature=0.2,
    )
    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("The language model returned an empty answer.")
    return answer.strip()
