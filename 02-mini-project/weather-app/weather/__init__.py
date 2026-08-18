"""Weather app: Open-Meteo client, domain model, and a small Flask front-end."""

from weather.client import LocationNotFound, WeatherClient, WeatherError
from weather.models import Forecast, Location

__all__ = ["WeatherClient", "WeatherError", "LocationNotFound", "Forecast", "Location"]
__version__ = "0.1.0"
