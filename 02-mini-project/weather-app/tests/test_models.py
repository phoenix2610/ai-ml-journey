from datetime import date, datetime

import pytest

from weather.models import (
    CurrentWeather,
    DayForecast,
    Forecast,
    Location,
    bearing_to_compass,
    describe,
    icon_for,
)

from conftest import FORECAST_PAYLOAD


# --------------------------------------------------------------------- codes


@pytest.mark.parametrize(
    "code,text", [(0, "Clear sky"), (3, "Overcast"), (63, "Moderate rain"), (95, "Thunderstorm")]
)
def test_describe_known_codes(code, text):
    assert describe(code) == text


def test_unknown_code_does_not_crash():
    assert describe(999) == "Unknown"
    assert icon_for(999) == "cloud"


def test_related_codes_share_an_icon():
    assert icon_for(61) == icon_for(63) == icon_for(65) == "rain"


@pytest.mark.parametrize(
    "degrees,point",
    [(0, "N"), (90, "E"), (180, "S"), (270, "W"), (45, "NE"), (350, "N"), (360, "N")],
)
def test_bearing_to_compass(degrees, point):
    assert bearing_to_compass(degrees) == point


# ------------------------------------------------------------------ Location


def test_label_joins_available_parts():
    loc = Location("Pune", 18.5, 73.8, country="India", admin1="Maharashtra")
    assert loc.label == "Pune, Maharashtra, India"


def test_label_skips_missing_parts():
    assert Location("Atlantis", 0.0, 0.0).label == "Atlantis"


def test_location_from_api():
    loc = Location.from_api(
        {"name": "Pune", "latitude": 18.5, "longitude": 73.8, "country": "India"}
    )
    assert (loc.name, loc.latitude, loc.country) == ("Pune", 18.5, "India")


# ------------------------------------------------------------------ Forecast


@pytest.fixture
def forecast():
    loc = Location("Pune", 18.51957, 73.85535, country="India", admin1="Maharashtra")
    return Forecast.from_api(loc, FORECAST_PAYLOAD)


def test_current_is_parsed(forecast):
    c = forecast.current
    assert c.temperature == 27.4
    assert c.humidity == 68
    assert c.time == datetime(2026, 8, 5, 12, 0)


def test_current_derives_description_and_compass(forecast):
    assert forecast.current.description == "Moderate rain"
    assert forecast.current.wind_compass == "WSW"


def test_daily_length_matches_payload(forecast):
    assert len(forecast.daily) == 3


def test_daily_entry_is_parsed(forecast):
    day = forecast.daily[0]
    assert day.day == date(2026, 8, 5)
    assert (day.temp_max, day.temp_min) == (29.0, 22.1)
    assert day.precipitation_chance == 90


def test_sunrise_keeps_only_the_time(forecast):
    assert forecast.daily[0].sunrise == "06:15"


def test_weekday_is_derived(forecast):
    assert forecast.daily[0].weekday == "Wed"


def test_units_are_carried_through(forecast):
    assert forecast.units["temperature"] == "°C"
    assert forecast.units["wind"] == "km/h"


def test_null_precipitation_probability_becomes_zero():
    payload = {**FORECAST_PAYLOAD}
    payload["daily"] = {**payload["daily"], "precipitation_probability_max": [None, 20, 0]}
    f = Forecast.from_api(Location("X", 0, 0), payload)
    assert f.daily[0].precipitation_chance == 0


def test_to_dict_is_json_shaped(forecast):
    data = forecast.to_dict()
    assert data["location"]["label"] == "Pune, Maharashtra, India"
    assert data["current"]["description"] == "Moderate rain"
    assert data["current"]["wind_direction"] == "WSW"
    assert len(data["daily"]) == 3
    assert data["daily"][0]["date"] == "2026-08-05"
