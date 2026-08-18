import pytest

from weather.client import LocationNotFound, TTLCache, WeatherClient, WeatherError

from conftest import FakeHttp


# ---------------------------------------------------------------------- cache


def test_ttl_cache_returns_stored_value():
    cache = TTLCache(10, clock=lambda: 0)
    cache.put("k", "v")
    assert cache.get("k") == "v"


def test_ttl_cache_miss_is_none():
    assert TTLCache(10).get("absent") is None


def test_ttl_cache_expires():
    now = [0.0]
    cache = TTLCache(10, clock=lambda: now[0])
    cache.put("k", "v")
    now[0] = 11
    assert cache.get("k") is None


def test_expired_entry_is_evicted():
    now = [0.0]
    cache = TTLCache(10, clock=lambda: now[0])
    cache.put("k", "v")
    now[0] = 11
    cache.get("k")
    assert len(cache) == 0


# ------------------------------------------------------------------ geocoding


def test_search_returns_locations(client):
    results = client.search("Pune")
    assert results[0].name == "Pune"
    assert results[0].country == "India"


def test_blank_query_rejected_without_a_call(client, http):
    with pytest.raises(LocationNotFound, match="enter a place"):
        client.search("   ")
    assert http.calls == []


def test_no_results_raises(client, http):
    http.geocode = {"results": []}
    with pytest.raises(LocationNotFound, match="no place called"):
        client.search("Xyzzy")


def test_null_results_key_is_handled(client, http):
    http.geocode = {"results": None}
    with pytest.raises(LocationNotFound):
        client.search("Xyzzy")


def test_geocode_results_are_cached(client, http):
    client.search("Pune")
    client.search("Pune")
    assert len([c for c in http.calls if "geocoding" in c[0]]) == 1
    assert client.stats["cache_hits"] == 1


def test_geocode_cache_is_case_insensitive(client, http):
    client.search("Pune")
    client.search("pune")
    assert len(http.calls) == 1


# ------------------------------------------------------------------- forecast


def test_for_place_geocodes_then_forecasts(client, http):
    forecast = client.for_place("Pune")
    assert forecast.location.name == "Pune"
    assert forecast.current.temperature == 27.4
    assert len(http.calls) == 2


def test_forecast_is_cached_by_coordinates(client, http):
    client.for_place("Pune")
    client.for_place("Pune")
    forecast_calls = [c for c in http.calls if "forecast" in c[0]]
    assert len(forecast_calls) == 1


def test_forecast_requests_the_fields_it_parses(client, http):
    client.for_place("Pune")
    params = [c[1] for c in http.calls if "forecast" in c[0]][0]
    assert "temperature_2m" in params["current"]
    assert "precipitation_probability_max" in params["daily"]
    assert params["timezone"] == "auto"


def test_imperial_units_are_requested(client, http):
    client.for_place("Pune", units="imperial")
    params = [c[1] for c in http.calls if "forecast" in c[0]][0]
    assert params["temperature_unit"] == "fahrenheit"
    assert params["wind_speed_unit"] == "mph"


def test_days_is_clamped_to_the_api_maximum(client, http):
    client.for_place("Pune", days=99)
    params = [c[1] for c in http.calls if "forecast" in c[0]][0]
    assert params["forecast_days"] == 16


def test_days_has_a_floor(client, http):
    client.for_place("Pune", days=0)
    params = [c[1] for c in http.calls if "forecast" in c[0]][0]
    assert params["forecast_days"] == 1


def test_malformed_forecast_payload_is_wrapped(client, http):
    http.forecast = {"current": {}, "daily": {}}
    with pytest.raises(WeatherError, match="unexpected forecast payload"):
        client.for_place("Pune")


def test_transport_error_propagates_as_weather_error():
    client = WeatherClient(FakeHttp(error=WeatherError("service down")))
    with pytest.raises(WeatherError, match="service down"):
        client.for_place("Pune")


def test_stats_count_api_calls(client):
    client.for_place("Pune")
    assert client.stats["api_calls"] == 2
