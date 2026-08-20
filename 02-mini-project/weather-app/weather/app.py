"""Flask front-end.

An application factory rather than a module-level ``app``, so tests can build
an instance with a fake HTTP transport injected. That is the difference between
a test suite that runs offline in milliseconds and one that hits a real API.

Two surfaces over the same client:
  GET /             HTML
  GET /api/weather  JSON
"""

from __future__ import annotations

import os

from flask import Flask, jsonify, render_template, request

from weather.client import LocationNotFound, WeatherClient, WeatherError

DEFAULT_PLACE = "Pune"
MAX_DAYS = 16


def create_app(client: WeatherClient | None = None, **config) -> Flask:
    app = Flask(__name__)
    app.config.update(
        DEFAULT_PLACE=os.environ.get("DEFAULT_PLACE", DEFAULT_PLACE),
        JSON_SORT_KEYS=False,
    )
    app.config.update(config)

    # One client per app: its caches are the whole point, so it must outlive
    # the request that created it.
    weather = client or WeatherClient()

    def _units() -> str:
        return "imperial" if request.args.get("units") == "imperial" else "metric"

    def _days() -> int:
        try:
            return max(1, min(int(request.args.get("days", 7)), MAX_DAYS))
        except (TypeError, ValueError):
            return 7

    @app.get("/")
    def index():
        query = (request.args.get("q") or "").strip()
        units = _units()

        if not query:
            return render_template(
                "index.html", forecast=None, query="", units=units, error=None
            )

        try:
            forecast = weather.for_place(query, days=_days(), units=units)
        except LocationNotFound as exc:
            return render_template(
                "index.html", forecast=None, query=query, units=units, error=str(exc)
            ), 404
        except WeatherError as exc:
            return render_template(
                "index.html", forecast=None, query=query, units=units, error=str(exc)
            ), 502

        return render_template(
            "index.html", forecast=forecast, query=query, units=units, error=None
        )

    @app.get("/api/weather")
    def api_weather():
        query = (request.args.get("q") or "").strip()
        if not query:
            return jsonify(error="missing required parameter: q"), 400

        try:
            forecast = weather.for_place(query, days=_days(), units=_units())
        except LocationNotFound as exc:
            return jsonify(error=str(exc)), 404
        except WeatherError as exc:
            return jsonify(error=str(exc)), 502

        return jsonify(forecast.to_dict())

    @app.get("/api/search")
    def api_search():
        query = (request.args.get("q") or "").strip()
        if not query:
            return jsonify(error="missing required parameter: q"), 400
        try:
            places = weather.search(query)
        except LocationNotFound as exc:
            return jsonify(error=str(exc)), 404
        except WeatherError as exc:
            return jsonify(error=str(exc)), 502
        return jsonify(
            results=[
                {"label": p.label, "latitude": p.latitude, "longitude": p.longitude}
                for p in places
            ]
        )

    @app.get("/healthz")
    def healthz():
        """Liveness probe for the platform. Deliberately does no I/O."""
        return jsonify(status="ok", cache=weather.stats)

    @app.errorhandler(404)
    def not_found(_e):
        if request.path.startswith("/api/"):
            return jsonify(error="not found"), 404
        return render_template(
            "index.html", forecast=None, query="", units="metric",
            error="That page does not exist.",
        ), 404

    return app


# Gunicorn entry point: `gunicorn 'weather.app:app'`
app = create_app()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
