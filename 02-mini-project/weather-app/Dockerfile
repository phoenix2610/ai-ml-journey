# Two stages: wheels get built with the toolchain, then copied into a runtime
# image that never had a compiler in it. Smaller, and a smaller attack surface.

FROM python:3.12-slim AS build

WORKDIR /build
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


FROM python:3.12-slim

# Do not run as root, and do not buffer logs the platform is trying to collect.
RUN useradd --create-home --uid 1000 app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

COPY --from=build /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY weather/ ./weather/

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,os,sys; \
    sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz', timeout=2).status==200 else 1)"

# Two workers with threads: this app is I/O-bound on an upstream API, so
# threads buy more than extra processes would on a small instance.
CMD gunicorn "weather.app:app" \
    --bind "0.0.0.0:${PORT}" \
    --workers 2 --threads 4 \
    --timeout 30 \
    --access-logfile - --error-logfile -
