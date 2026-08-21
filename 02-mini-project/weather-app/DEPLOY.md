# Deploying

The app needs **no API key and no database**, which is what makes a free tier
genuinely viable here. Open-Meteo is free for non-commercial use and
unauthenticated, so there are no secrets to manage.

## Locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

flask --app weather.app run --debug        # http://127.0.0.1:5000
```

## Docker

```bash
docker build -t weather-app .
docker run --rm -p 8000:8000 weather-app
curl localhost:8000/healthz
```

The build is two-stage: wheels are compiled in an image that has a toolchain,
then copied into a runtime image that never did. That keeps the final image
small and removes the compiler from the shipped attack surface. The container
runs as UID 1000, not root.

## Render (recommended free option)

`render.yaml` is a blueprint, so the deployment is version-controlled rather
than clicked together in a dashboard.

1. Push this directory to GitHub.
2. Go to **Dashboard → Blueprints → New Blueprint Instance**.
3. Point it at the repo. Render reads `render.yaml` and provisions the service.
4. First build takes ~3 minutes; subsequent pushes auto-deploy.

Free instances **sleep after 15 minutes of inactivity** and take ~30 seconds to
wake. That is fine for a portfolio link, and the `/healthz` endpoint gives an
uptime pinger something cheap to hit if you want to keep it warm.

## Fly.io

```bash
fly launch --no-deploy       # generates fly.toml from the Dockerfile
fly deploy
```

Set `internal_port = 8000` and `force_https = true` in the generated
`fly.toml`. Fly's free allowance covers a single shared-cpu-1x instance.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8000` | Port gunicorn binds. Most platforms inject this. |
| `DEFAULT_PLACE` | `Pune` | Placeholder shown before a search. |
| `FLASK_DEBUG` | unset | `1` enables the reloader. **Never set in production.** |

## Notes for a real deployment

**Caching is per-process.** Forecasts are cached in memory with a 10-minute
TTL, so with two gunicorn workers you get two independent caches — a cache hit
rate of roughly 50% rather than 100%. That is an acceptable trade at this
scale; a shared Redis would fix it and is the obvious first change if traffic
ever justified it.

**Worker model.** Two workers with four threads each. The app is I/O-bound
waiting on an upstream HTTP call, so threads buy more concurrency per MB than
additional processes would on a small instance.

**Rate limits.** Open-Meteo asks for under 10,000 requests/day for the free
tier. The TTL cache keeps a normal portfolio-traffic app far under that; if you
expect real traffic, raise `forecast_ttl` before doing anything else.
