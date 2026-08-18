"""Shared fixtures: a fake Open-Meteo that never touches the network."""

import pytest

from weather.client import WeatherClient

GEOCODE_PAYLOAD = {
    "results": [
        {
            "name": "Pune",
            "latitude": 18.51957,
            "longitude": 73.85535,
            "country": "India",
            "admin1": "Maharashtra",
            "timezone": "Asia/Kolkata",
        }
    ]
}

FORECAST_PAYLOAD = {
    "current_units": {"temperature_2m": "°C", "wind_speed_10m": "km/h"},
    "daily_units": {"precipitation_sum": "mm"},
    "current": {
        "time": "2026-08-05T12:00",
        "temperature_2m": 27.4,
        "apparent_temperature": 29.1,
        "relative_humidity_2m": 68,
        "precipitation": 0.2,
        "weather_code": 63,
        "wind_speed_10m": 14.8,
        "wind_direction_10m": 250,
        "is_day": 1,
    },
    "daily": {
        "time": ["2026-08-05", "2026-08-06", "2026-08-07"],
        "weather_code": [63, 3, 0],
        "temperature_2m_max": [29.0, 30.2, 31.5],
        "temperature_2m_min": [22.1, 22.8, 23.0],
        "precipitation_sum": [12.4, 0.0, 0.0],
        "precipitation_probability_max": [90, 20, 0],
        "sunrise": ["2026-08-05T06:15", "2026-08-06T06:15", "2026-08-07T06:16"],
        "sunset": ["2026-08-05T19:02", "2026-08-06T19:01", "2026-08-07T19:00"],
    },
}


class FakeHttp:
    """Routes on URL, records every call, and can be told to fail."""

    def __init__(self, geocode=None, forecast=None, error=None):
        self.geocode = GEOCODE_PAYLOAD if geocode is None else geocode
        self.forecast = FORECAST_PAYLOAD if forecast is None else forecast
        self.error = error
        self.calls = []

    def get_json(self, url, params):
        self.calls.append((url, params))
        if self.error:
            raise self.error
        return self.geocode if "geocoding" in url else self.forecast


@pytest.fixture
def http():
    return FakeHttp()


@pytest.fixture
def client(http):
    return WeatherClient(http)
