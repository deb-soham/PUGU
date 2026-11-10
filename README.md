# PUGU: Personalized Understanding and Guidance Unit

A free, open-source web app with two tools:

1. **Focus (ADHD):** breaks a task into small steps and shows them one at a time.
2. **Read (Dyslexia):** simplifies pasted text and shows it with spacing controls and
   read-aloud that highlights the current sentence and pauses between sentences.

PUGU supports focus and reading. It is not a medical tool and does not diagnose or treat anything.

## Tech stack
Streamlit, Google sign-in, Supabase (database), Groq (AI). All have free tiers.

## Run it locally
1. Install Python 3.10+ and Git.
2. Clone the repo and create a virtual environment:
   `python -m venv venv`, then activate it, then `pip install -r requirements.txt`
3. Create a free Supabase project and run `sql/schema.sql` in its SQL Editor.
4. Create a Google OAuth client (type: Web application) with the redirect URI
   `http://localhost:8501/oauth2callback`.
5. Get a free Groq API key.
6. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in your values.
   **Never commit `secrets.toml`.**
7. Run `streamlit run app.py`.

## Contributing
Fork the repo or create a branch, make a small focused change, and open a pull request.
Never commit secrets.

## License
MIT. See `LICENSE`.