# Deploying

## The one secret

`GEMINI_API_KEY`. Nothing else is required — no database, no vector service.

```bash
export GEMINI_API_KEY=your-key-here
```

**Do not put a space before the `=`.** `GEMINI_API_KEY =abc` gives the variable
the name `"GEMINI_API_KEY "` with a trailing space, and every standard lookup
misses it. `rag.llm.load_dotenv` strips whitespace and matches the name
case-insensitively (so `Gemini_API_KEY` also resolves), but most other tooling
will not, so it is worth writing it plainly.

## Locally

```bash
pip install -r requirements.txt
export RAG_DOCS=./docs          # any directory of .md files
streamlit run app.py            # http://localhost:8501
```

First run embeds the corpus and writes `index/` plus `.cache/embeddings.json`.
Later runs load the index directly. Editing one document re-embeds only the
chunks that changed, because the cache is keyed on a hash of the chunk text
plus the model name.

## Docker

```bash
docker build -t rag-chatbot .
docker run --rm -p 8501:8501 \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  -v "$PWD/docs:/app/docs:ro" \
  -v rag-index:/app/index \
  rag-chatbot
```

Mount `index` as a named volume so the container does not re-embed the whole
corpus on every restart.

## Streamlit Community Cloud (free)

1. Push to GitHub.
2. share.streamlit.io → **New app** → point at `app.py`.
3. **Settings → Secrets**:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
4. Deploy.

Streamlit secrets land in the environment, so `load_key()` finds them with no
code change. Free apps sleep after inactivity and take ~30s to wake.

## Rate limits — read this before demoing

The Gemini free tier allows roughly **5 `generateContent` requests per minute**
for `gemini-3.6-flash`. That is low enough to break a live demo, and it is how
this project found the limit: a 12-question evaluation run failed partway with
HTTP 429.

Two mitigations, both already implemented:

**The client honours the server's own retry hint.** A 429 carries the exact
delay to wait (`retryDelay` in `error.details`, and in the message text).
Backing off by that amount beats exponential backoff, which either sleeps far
too long or hammers the limit again.

**Client-side pacing.** Set a minimum gap between generation calls:

```bash
export RAG_MIN_INTERVAL=13      # seconds; ~4.6 requests/minute
```

Leave it at `0` for interactive use — a human cannot type fast enough to hit
the limit. Set it before running the evaluation, which fires 12 requests back
to back.

Embeddings have a much more generous quota, so ingestion is not the constraint.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | **Required.** |
| `RAG_DOCS` | `docs` | Directory of markdown to index. |
| `RAG_MIN_INTERVAL` | `0` | Seconds between generation calls. |
| `PORT` | `8501` | Streamlit port. |

## Cost

Retrieval-augmented answers are small: ~380 prompt tokens and ~20 output tokens
per question, measured over the evaluation run. Flash pricing makes a few
thousand questions negligible, and embeddings are a one-off per document
version thanks to the cache.

The `Usage` counter in the sidebar reports real per-session numbers rather than
estimates.

## Before shipping this publicly

- **Add auth or a rate limit.** A public URL with your key behind it is a
  public URL that spends your quota.
- **Watch the index directory.** It is written at runtime; on an ephemeral
  filesystem the app re-embeds on every cold start.
- **Re-run `evaluate` after changing chunk size, `top_k`, or the retrieval
  strategy.** Those three change retrieval quality more than any prompt edit,
  and the golden set is what tells you which direction you moved.
