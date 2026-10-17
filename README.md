# Banking Information Assistant

An original Streamlit demo for general banking education. It retrieves relevant
articles from a small, bundled FAQ and can optionally use an OpenAI model to
write a concise answer grounded in those excerpts.

## Features

- Chat-style Streamlit interface with reference excerpts shown alongside answers.
- Dependency-free TF-IDF retrieval, available in local demo mode without API credentials.
- Optional OpenAI answer generation using an environment variable or Streamlit secret.
- Safety boundaries: no personalized financial advice, account access, or transactions.

## Run locally

Requires Python 3.10 or later.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Without `OPENAI_API_KEY`, the app returns matching excerpts only. To enable
generated answers, set the key in your shell before starting Streamlit:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
# Optional; defaults to gpt-4o-mini
$env:OPENAI_MODEL = "gpt-4o-mini"
streamlit run app.py
```

Never commit API keys. `.env.example` documents the optional settings; the
application reads environment variables or Streamlit deployment secrets and
does not store credentials in the repository.

### Streamlit Community Cloud

After deploying the repository, open the app's **Settings → Secrets** and add:

```toml
OPENAI_API_KEY = "your-api-key"
OPENAI_MODEL = "gpt-4o-mini"
```

Save the secrets and restart/reboot the app. Add the key only in the private
Secrets settings—not in a source file, commit, or chat message.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## Safety and scope

This is an educational prototype, not a bank service or a source of personalized
financial, legal, or tax advice. The included reference library is intentionally
small and may not reflect current or locally applicable rules. Verify important
information with official institutions and regulators. Never enter account
numbers, passwords, PINs, or one-time verification codes. The app cannot access
accounts or initiate transactions.
