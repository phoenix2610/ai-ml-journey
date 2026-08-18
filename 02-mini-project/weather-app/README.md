# Weather App

A seven-day forecast for anywhere, served from a Flask app with **no API key,
no database, and no build step**. Open-Meteo is free and unauthenticated, so
there is nothing to rotate and nothing to leak — which is exactly why it can
sit on a free tier indefinitely.

```
GET /                      HTML forecast
GET /api/weather?q=Pune    same data as JSON
GET /api/search?q=Pun      geocoding autocomplete
GET /healthz               liveness probe
```

## Run it

```bash
pip install -r requirements.txt
flask --app weather.app run --debug     # http://127.0.0.1:5000
pytest -q                               # 65 tests, no network
```

See [DEPLOY.md](./DEPLOY.md) for Docker, Render, and Fly.

## The one design decision that mattered

Everything else follows from **injecting the HTTP transport**:

```python
class Http(Protocol):
    def get_json(self, url: str, params: dict) -> dict: ...

def create_app(client: WeatherClient | None = None) -> Flask: ...
```

`WeatherClient` depends on that protocol, and the app is built by a factory
that accepts a client. So the tests construct an app wired to a fake that
replays canned Open-Meteo payloads, and the **entire 65-test suite runs offline
in 0.25 seconds** — including the route tests.

The alternative — a module-level `app` and a hard dependency on `requests` —
means either mocking at the library level or hitting a live API in CI. Both are
worse, and the second one is flaky by construction.

## Caching

Two caches with very different TTLs, because they answer different questions:

| | TTL | Why |
|---|---|---|
| Geocoding | 24 hours | Pune is not going to move. |
| Forecasts | 10 minutes | Long enough to absorb a refresh, short enough to stay true. |

Cache keys for forecasts are rounded coordinates, not the search string, so
"Pune", "pune", and "Pune, India" all share one entry.

The client is created **once per app**, not per request — its caches are the
entire point, so it has to outlive the request that created it. There is a test
pinning that, because it is an easy thing to break later.

## Reading the data

The API answers in integers: `weather_code: 63`. The WMO 4677 lookup lives in
`models.py` rather than in a template, because the HTML view, the JSON
endpoint, and the tests all need the same mapping.

```python
63  -> ("Moderate rain", "rain")
95  -> ("Thunderstorm",  "storm")
250 -> wind_compass "WSW"        # 16-point, wraps correctly past 348.75°
```

Unknown codes degrade to `("Unknown", "cloud")` instead of raising — a forecast
missing one label is still a useful forecast.

## Errors are HTTP statuses, not stack traces

| Situation | HTML | JSON |
|---|---|---|
| Place not found | 404 + message in the page | 404 `{"error": ...}` |
| Upstream down | 502 + message in the page | 502 `{"error": ...}` |
| Missing `q` | renders the empty state | 400 `{"error": ...}` |

`urllib` exceptions are caught at the transport boundary and re-raised as
`WeatherError` with text meant for a person: *"could not reach the weather
service: Name or service not known"* rather than an `URLError` traceback.

## Front-end

One stylesheet, no framework, no JavaScript. Light and dark both come from a
single set of custom properties under `prefers-color-scheme`. The current-
conditions card gets a soft gradient wash keyed to the weather code, and the
daily rows use small coloured squares instead of an icon font — so the page
makes **zero external requests** and renders instantly.

## Layout

```
weather/
├── client.py     Open-Meteo calls, TTL caches, error normalisation
├── models.py     Location/Forecast types + the WMO code table
├── app.py        Flask factory, HTML + JSON routes
├── templates/    Jinja2, two files
└── static/       one stylesheet
```
