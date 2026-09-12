# Wheels are built in an image with a toolchain, then copied into a runtime
# image that never had one.

FROM python:3.12-slim AS build

WORKDIR /build
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


FROM python:3.12-slim

RUN useradd --create-home --uid 1000 app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501 \
    RAG_DOCS=/app/docs

WORKDIR /app

COPY --from=build /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY rag/ ./rag/
COPY eval/ ./eval/
COPY app.py .

# The index and embedding cache are written at runtime, so they need to be
# writable by the non-root user.
RUN mkdir -p /app/index /app/.cache /app/docs && chown -R app:app /app

USER app
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,os,sys; \
    sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/_stcore/health', timeout=3).status==200 else 1)"

CMD streamlit run app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
