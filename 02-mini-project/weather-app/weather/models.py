"""Domain types, and the WMO code table that makes a forecast readable.

The API answers with integers: `weather_code: 63` means "moderate rain". That
lookup lives here rather than in a template, because the same mapping is needed
by the HTML view, the JSON endpoint, and the tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

# WMO 4677 present-weather codes, grouped as Open-Meteo emits them.
WMO_CODES: dict[int, tuple[str, str]] = {
    0: ("Clear sky", "sun"),
    1: ("Mainly clear", "sun-cloud"),
    2: ("Partly cloudy", "sun-cloud"),
    3: ("Overcast", "cloud"),
    45: ("Fog", "fog"),
    48: ("Depositing rime fog", "fog"),
    51: ("Light drizzle", "drizzle"),
    53: ("Moderate drizzle", "drizzle"),
    55: ("Dense drizzle", "drizzle"),
    56: ("Light freezing drizzle", "sleet"),
    57: ("Dense freezing drizzle", "sleet"),
    61: ("Slight rain", "rain"),
    63: ("Moderate rain", "rain"),
    65: ("Heavy rain", "rain"),
    66: ("Light freezing rain", "sleet"),
    67: ("Heavy freezing rain", "sleet"),
    71: ("Slight snow", "snow"),
    73: ("Moderate snow", "snow"),
    75: ("Heavy snow", "snow"),
    77: ("Snow grains", "snow"),
    80: ("Slight rain showers", "rain"),
    81: ("Moderate rain showers", "rain"),
    82: ("Violent rain showers", "rain"),
    85: ("Slight snow showers", "snow"),
    86: ("Heavy snow showers", "snow"),
    95: ("Thunderstorm", "storm"),
    96: ("Thunderstorm with slight hail", "storm"),
    99: ("Thunderstorm with heavy hail", "storm"),
}

COMPASS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")


def describe(code: int) -> str:
    return WMO_CODES.get(code, ("Unknown", "cloud"))[0]


def icon_for(code: int) -> str:
    return WMO_CODES.get(code, ("Unknown", "cloud"))[1]


def bearing_to_compass(degrees: float) -> str:
    """0-360 -> one of 16 points. 348.75+ wraps back to N."""
    return COMPASS[round(degrees / 22.5) % 16]


@dataclass(frozen=True)
class Location:
    name: str
    latitude: float
    longitude: float
    country: str = ""
    admin1: str = ""          # state / region
    timezone: str = "UTC"

    @property
    def label(self) -> str:
        """'Pune, Maharashtra, India' -- skipping the parts the API omitted."""
        return ", ".join(p for p in (self.name, self.admin1, self.country) if p)

    @classmethod
    def from_api(cls, raw: dict) -> "Location":
        return cls(
            name=raw.get("name", "Unknown"),
            latitude=float(raw["latitude"]),
            longitude=float(raw["longitude"]),
            country=raw.get("country", ""),
            admin1=raw.get("admin1", ""),
            timezone=raw.get("timezone", "UTC"),
        )


@dataclass(frozen=True)
class CurrentWeather:
    time: datetime
    temperature: float
    feels_like: float
    humidity: int
    precipitation: float
    wind_speed: float
    wind_direction: int
    is_day: bool
    code: int

    @property
    def description(self) -> str:
        return describe(self.code)

    @property
    def icon(self) -> str:
        return icon_for(self.code)

    @property
    def wind_compass(self) -> str:
        return bearing_to_compass(self.wind_direction)


@dataclass(frozen=True)
class DayForecast:
    day: date
    code: int
    temp_max: float
    temp_min: float
    precipitation: float
    precipitation_chance: int
    sunrise: str
    sunset: str

    @property
    def description(self) -> str:
        return describe(self.code)

    @property
    def icon(self) -> str:
        return icon_for(self.code)

    @property
    def weekday(self) -> str:
        return self.day.strftime("%a")


@dataclass(frozen=True)
class Forecast:
    location: Location
    current: CurrentWeather
    daily: list[DayForecast]
    units: dict[str, str]

    @classmethod
    def from_api(cls, location: Location, raw: dict) -> "Forecast":
        current = raw["current"]
        daily = raw["daily"]

        return cls(
            location=location,
            current=CurrentWeather(
                time=datetime.fromisoformat(current["time"]),
                temperature=current["temperature_2m"],
                feels_like=current["apparent_temperature"],
                humidity=int(current["relative_humidity_2m"]),
                precipitation=current.get("precipitation", 0.0),
                wind_speed=current["wind_speed_10m"],
                wind_direction=int(current.get("wind_direction_10m", 0)),
                is_day=bool(current.get("is_day", 1)),
                code=int(current["weather_code"]),
            ),
            daily=[
                DayForecast(
                    day=date.fromisoformat(daily["time"][i]),
                    code=int(daily["weather_code"][i]),
                    temp_max=daily["temperature_2m_max"][i],
                    temp_min=daily["temperature_2m_min"][i],
                    precipitation=daily["precipitation_sum"][i],
                    precipitation_chance=int(daily["precipitation_probability_max"][i] or 0),
                    sunrise=daily["sunrise"][i].split("T")[-1],
                    sunset=daily["sunset"][i].split("T")[-1],
                )
                for i in range(len(daily["time"]))
            ],
            units={
                "temperature": raw.get("current_units", {}).get("temperature_2m", "°C"),
                "wind": raw.get("current_units", {}).get("wind_speed_10m", "km/h"),
                "precipitation": raw.get("daily_units", {}).get("precipitation_sum", "mm"),
            },
        )

    def to_dict(self) -> dict:
        """Serialisable form for the JSON endpoint."""
        return {
            "location": {
                "label": self.location.label,
                "latitude": self.location.latitude,
                "longitude": self.location.longitude,
                "timezone": self.location.timezone,
            },
            "current": {
                "time": self.current.time.isoformat(),
                "temperature": self.current.temperature,
                "feels_like": self.current.feels_like,
                "humidity": self.current.humidity,
                "wind_speed": self.current.wind_speed,
                "wind_direction": self.current.wind_compass,
                "description": self.current.description,
                "icon": self.current.icon,
                "is_day": self.current.is_day,
            },
            "daily": [
                {
                    "date": d.day.isoformat(),
                    "weekday": d.weekday,
                    "temp_max": d.temp_max,
                    "temp_min": d.temp_min,
                    "precipitation": d.precipitation,
                    "precipitation_chance": d.precipitation_chance,
                    "description": d.description,
                    "icon": d.icon,
                    "sunrise": d.sunrise,
                    "sunset": d.sunset,
                }
                for d in self.daily
            ],
            "units": self.units,
        }


__all__ = [
    "Location", "CurrentWeather", "DayForecast", "Forecast",
    "describe", "icon_for", "bearing_to_compass", "WMO_CODES",
]
