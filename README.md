# Banking Guide AI

An original, small retrieval-augmented banking education demo built with Streamlit. It retrieves relevant passages from a local guide using TF-IDF search. If an OpenAI API key is configured, it can use a language model to write a concise answer from those passages; without a key, it displays the retrieved passages directly.

This is an independent project and is not affiliated with a bank or based on copied code from another repository.

## What it includes

- Streamlit chat interface and source references
- Local TF-IDF retrieval over the included educational banking guide
- Optional OpenAI answer generation; no key is needed for local retrieval
- Basic safety boundaries for account access and transactions
- Unit tests for retrieval behavior

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the virtual environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the app in local retrieval mode:

```bash
streamlit run app.py
```

To enable generated answers, set `OPENAI_API_KEY` in your environment. You can optionally set `OPENAI_MODEL` (defaults to `gpt-4o-mini`). Never commit real API keys. `.env.example` documents the supported variables.

Run the retrieval tests:

```bash
python -m unittest discover -s tests
```

## Safety and limitations

This is an educational demo, not a banking service. It cannot access accounts, check balances, or perform transactions. Do not enter passwords, verification codes, card numbers, or account details. The guide is illustrative and does not replace current disclosures from a bank, regulator, or deposit insurer. AI-generated text can be incorrect; verify important information using official sources. This app does not provide personalized financial, legal, or tax advice.
