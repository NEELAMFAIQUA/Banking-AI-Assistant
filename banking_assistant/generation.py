"""Optional grounded answer generation; retrieval works without an API key."""

from __future__ import annotations

import os
from typing import Any

SYSTEM_PROMPT = """You are a banking information assistant for general education.
Answer only using the supplied reference excerpts. If they do not contain the
answer, say that you cannot determine it from the available information.
Keep answers concise. Do not give personalized financial, legal, or tax advice.
Never request or process passwords, PINs, full account numbers, or other
authentication secrets. You cannot access accounts or perform transactions.
Encourage users to contact their bank through its official channels for
account-specific questions."""


def llm_is_configured() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def generate_answer(question: str, articles: list[dict[str, Any]]) -> str:
    """Generate an answer grounded in retrieved articles using OpenAI."""
    from openai import OpenAI

    excerpts = "\n\n".join(
        f"Source: {article['source']}\n"
        f"Topic: {article['title']}\n"
        f"Reference: {article['answer']}"
        for article in articles
    )
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
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
