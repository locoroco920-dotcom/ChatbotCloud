# Meadowlands Chatbot

FastAPI chatbot with semantic FAQ matching, ready for Render deployment.

## Environment Variables

Set these in your shell (local) or Render dashboard:

- `OPENAI_API_KEY` - OpenAI key (optional in current FAQ flow, but initialized safely)
- `WIDGET_API_KEY` - required for `POST /chat` authentication via `x-api-key` header
- `FRONTEND_ORIGINS` - comma-separated origins for CORS (example: `https://app.example.com,https://www.example.com`)

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /health` → `{ "ok": true }`
- `POST /ask` → existing chatbot response schema (no API key required)
- `POST /chat` → same response schema, requires `x-api-key` matching `WIDGET_API_KEY`

Request body for `/ask` and `/chat`:

```json
{
  "question": "Where can I eat?"
}
```

Response schema:

```json
{
  "answer": "Try Vesta Wood Fired for pizza or Candlewyck Diner for late-night dining.",
  "confidence": 0.95
}
```

## Render Deployment

1. Push this repo to GitHub.
2. In Render, create a new Web Service from the repo.
3. Render will detect `render.yaml`.
4. Ensure env vars are set in Render:
   - `OPENAI_API_KEY`
   - `WIDGET_API_KEY`
   - `FRONTEND_ORIGINS`
5. Deploy. Start command is:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

## Development-only Extras

For local tunnel debugging only:

```bash
pip install -r requirements-dev.txt
```

`requirements-dev.txt` includes `pyngrok`, while production `requirements.txt` does not.

## Build FAQ from Full Website

To make the chatbot understand the full MLCVB site (events, restaurants, directories, services), rebuild `faq_data.json` by crawling the site:

```bash
pip install -r requirements-dev.txt
python build_site_faq.py --start-url https://dev.mlcvb.com/ --max-pages 220 --output faq_data.json
```

Then redeploy (or restart locally). The generated answers include source links so the bot can point users directly to relevant pages.
