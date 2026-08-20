import re

import pytest

from weather.app import create_app
from weather.client import WeatherClient, WeatherError

from conftest import FakeHttp


@pytest.fixture
def app(client):
    application = create_app(client)
    application.config.update(TESTING=True)
    return application


@pytest.fixture
def web(app):
    return app.test_client()


# ------------------------------------------------------------------ HTML view


def test_home_renders_without_a_query(web):
    res = web.get("/")
    assert res.status_code == 200
    assert b"Search for a place" in res.data


def test_search_renders_a_forecast(web):
    res = web.get("/?q=Pune")
    assert res.status_code == 200
    body = res.data.decode()
    assert "Pune, Maharashtra, India" in body
    assert "Moderate rain" in body
    assert "27" in body                     # rounded current temperature


def test_forecast_lists_every_day(web):
    body = web.get("/?q=Pune").data.decode()
    assert body.count('class="day"') == 3


def test_unknown_place_renders_an_error_with_404(web, http):
    http.geocode = {"results": []}
    res = web.get("/?q=Xyzzy")
    assert res.status_code == 404
    assert b"no place called" in res.data


def test_upstream_failure_renders_502(app, http):
    http.error = WeatherError("service down")
    res = app.test_client().get("/?q=Pune")
    assert res.status_code == 502
    assert b"service down" in res.data


@pytest.mark.parametrize("chosen,other", [("imperial", "metric"), ("metric", "imperial")])
def test_units_toggle_marks_only_the_active_unit(web, chosen, other):
    body = web.get(f"/?q=Pune&units={chosen}").data.decode()
    classes = dict(re.findall(r'<a href="[^"]*units=(\w+)"\s+class="([^"]*)"', body))
    assert classes[chosen] == "on"
    assert classes[other] == ""


def test_imperial_units_reach_the_api(app, http):
    app.test_client().get("/?q=Pune&units=imperial")
    params = [c[1] for c in http.calls if "forecast" in c[0]][0]
    assert params["temperature_unit"] == "fahrenheit"


def test_blank_query_is_not_an_error(web):
    assert web.get("/?q=%20%20").status_code == 200


def test_unknown_page_returns_404(web):
    assert web.get("/nope").status_code == 404


# ------------------------------------------------------------------ JSON API


def test_api_returns_json(web):
    res = web.get("/api/weather?q=Pune")
    assert res.status_code == 200
    data = res.get_json()
    assert data["location"]["label"] == "Pune, Maharashtra, India"
    assert data["current"]["description"] == "Moderate rain"
    assert len(data["daily"]) == 3


def test_api_requires_q(web):
    res = web.get("/api/weather")
    assert res.status_code == 400
    assert "missing required parameter" in res.get_json()["error"]


def test_api_unknown_place_is_404(web, http):
    http.geocode = {"results": []}
    res = web.get("/api/weather?q=Xyzzy")
    assert res.status_code == 404
    assert "error" in res.get_json()


def test_api_upstream_failure_is_502(app, http):
    http.error = WeatherError("nope")
    res = app.test_client().get("/api/weather?q=Pune")
    assert res.status_code == 502


def test_api_404_on_unknown_api_path_is_json(web):
    res = web.get("/api/nothing")
    assert res.status_code == 404
    assert res.get_json()["error"] == "not found"


def test_search_endpoint(web):
    data = web.get("/api/search?q=Pune").get_json()
    assert data["results"][0]["label"].startswith("Pune")


def test_days_parameter_is_clamped(app, http):
    app.test_client().get("/api/weather?q=Pune&days=999")
    params = [c[1] for c in http.calls if "forecast" in c[0]][0]
    assert params["forecast_days"] == 16


def test_garbage_days_parameter_falls_back(app, http):
    app.test_client().get("/api/weather?q=Pune&days=banana")
    params = [c[1] for c in http.calls if "forecast" in c[0]][0]
    assert params["forecast_days"] == 7


# -------------------------------------------------------------------- health


def test_healthz(web):
    res = web.get("/healthz")
    assert res.status_code == 200
    assert res.get_json()["status"] == "ok"


def test_healthz_does_no_upstream_io(web, http):
    web.get("/healthz")
    assert http.calls == []


def test_client_is_shared_across_requests(web, http):
    # Second request must be served from the client's cache, not a new call.
    web.get("/?q=Pune")
    web.get("/?q=Pune")
    assert len([c for c in http.calls if "forecast" in c[0]]) == 1
