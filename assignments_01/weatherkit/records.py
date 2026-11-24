from dataclasses import dataclass

from weatherkit.schemas import WeatherResponse

# HourlyReading is a dataclass because the data is already trusted by the
# time it gets here. The boundary is where the API data comes into my
# program, and WeatherResponse in schemas.py checks it there (types, ranges,
# list lengths). After that check, I only need a simple record to hold the
# values, so I don't need to validate them again. A dataclass is lighter and
# simpler for that.

@dataclass
class HourlyReading:
    """One hour of weather.

    Attributes:
        timestamp: Local time, e.g. "2026-04-08T14:00".
        temperature_c: Air temperature in degrees Celsius.
        precipitation_mm: Rainfall for that hour in millimeters.
    """
    timestamp: str
    temperature_c: float
    precipitation_mm: float


def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Turn the columnar hourly data into one reading per hour.

    Args:
        response: A validated WeatherResponse.

    Returns:
        A list of HourlyReading objects, in the same order as the input.
    """
    hourly = response.hourly
    readings = []
    # zip walks the three lists together: seat i of each train car
    for time, temp, precip in zip(
        hourly.time, hourly.temperature_2m, hourly.precipitation
    ):
        readings.append(HourlyReading(time, temp, precip))
    return readings