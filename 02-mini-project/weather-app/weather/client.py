"""Open-Meteo client.

Open-Meteo needs no API key, which removes the usual secret-management chore
and makes the deployed app genuinely free to run. Two calls are involved:
geocoding a place name to coordinates, then asking for the forecast.

Geocoding results are cached with a long TTL because "Pune" will not move.
Forecasts get a short TTL -- long enough to absorb a page refresh, short enough
that the number on screen is still true.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from weather.models import Forecast, Location

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

CURRENT_FIELDS = (
    "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,"
    "weather_code,wind_speed_10m,wind_direction_10m,is_day"
)
DAILY_FIELDS = (
    "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,"
    "precipitation_probability_max,sunrise,sunset"
)


class WeatherError(RuntimeError):
    """Anything that stops us producing a forecast, phrased for a end user."""


class LocationNotFound(WeatherError):
    pass


class Http(Protocol):
    def get_json(self, url: str, params: dict[str, Any]) -> dict: ...


class UrllibHttp:
    """Standard-library transport -- one less dependency in the container."""

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout

    def get_json(self, url: str, params: dict[str, Any]) -> dict:
        query = urllib.parse.urlencode(params)
        request = urllib.request.Request(
            f"{url}?{query}", headers={"User-Agent": "journey-weather/0.1"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            raise WeatherError(f"weather service returned HTTP {exc.code}") from None
        except urllib.error.URLError as exc:
            raise WeatherError(f"could not reach the weather service: {exc.reason}") from None
        except (json.JSONDecodeError, TimeoutError):
            raise WeatherError("weather service sent an unreadable response") from None


@dataclass
class _Entry:
    value: Any
    expires_at: float


class TTLCache:
    def __init__(self, ttl: float, *, clock=time.monotonic) -> None:
        self.ttl = ttl
        self._clock = clock
        self._data: dict[str, _Entry] = {}

    def get(self, key: str) -> Any | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        if self._clock() >= entry.expires_at:
            del self._data[key]
            return None
        return entry.value

    def put(self, key: str, value: Any) -> None:
        self._data[key] = _Entry(value, self._clock() + self.ttl)

    def __len__(self) -> int:
        return len(self._data)


class WeatherClient:
    def __init__(
        self,
        http: Http | None = None,
        *,
        forecast_ttl: float = 600,     # 10 minutes
        geocode_ttl: float = 86_400,   # a day; cities do not move
        clock=time.monotonic,
    ) -> None:
        self.http = http or UrllibHttp()
        self._forecasts = TTLCache(forecast_ttl, clock=clock)
        self._places = TTLCache(geocode_ttl, clock=clock)
        self.stats = {"api_calls": 0, "cache_hits": 0}

    # -------------------------------------------------------------- geocoding

    def search(self, query: str, *, limit: int = 5) -> list[Location]:
        query = query.strip()
        if not query:
            raise LocationNotFound("please enter a place name")

        key = f"{query.lower()}:{limit}"
        cached = self._places.get(key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached

        self.stats["api_calls"] += 1
        payload = self.http.get_json(
            GEOCODE_URL, {"name": query, "count": limit, "format": "json"}
        )

        results = [Location.from_api(r) for r in payload.get("results") or []]
        if not results:
            raise LocationNotFound(f"no place called {query!r} was found")

        self._places.put(key, results)
        return results

    def geocode(self, query: str) -> Location:
        return self.search(query, limit=1)[0]

    # --------------------------------------------------------------- forecast

    def forecast(self, location: Location, *, days: int = 7, units: str = "metric") -> Forecast:
        key = f"{location.latitude:.3f},{location.longitude:.3f}:{days}:{units}"
        cached = self._forecasts.get(key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached

        params: dict[str, Any] = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "current": CURRENT_FIELDS,
            "daily": DAILY_FIELDS,
            "timezone": "auto",
            "forecast_days": max(1, min(days, 16)),
        }
        if units == "imperial":
            params |= {
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "precipitation_unit": "inch",
            }

        self.stats["api_calls"] += 1
        payload = self.http.get_json(FORECAST_URL, params)

        try:
            result = Forecast.from_api(location, payload)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise WeatherError(f"unexpected forecast payload: {exc}") from None

        self._forecasts.put(key, result)
        return result

    def for_place(self, query: str, *, days: int = 7, units: str = "metric") -> Forecast:
        """Geocode then forecast -- what the web layer actually calls."""
        return self.forecast(self.geocode(query), days=days, units=units)


__all__ = [
    "WeatherClient", "WeatherError", "LocationNotFound",
    "TTLCache", "Http", "UrllibHttp",
]
