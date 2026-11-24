import pytest

from weatherkit import WeatherResponse, HourlyReading, to_readings


@pytest.fixture
def response():
    return WeatherResponse.model_validate({
        "latitude": 35.2,
        "longitude": -80.8,
        "timezone": "America/New_York",
        "elevation": 230.0,
        "hourly": {
            "time": ["2026-04-08T00:00", "2026-04-08T01:00", "2026-04-08T02:00"],
            "temperature_2m": [10.0, 11.5, 12.0],
            "precipitation": [0.0, 0.4, 1.2],
        },
    })


def test_one_reading_per_hour_in_order(response):
    readings = to_readings(response)
    assert len(readings) == 3
    assert readings[0].timestamp == "2026-04-08T00:00"
    assert readings[-1].timestamp == "2026-04-08T02:00"


def test_values_match_their_index(response):
    readings = to_readings(response)
    for i, reading in enumerate(readings):
        assert reading.timestamp == response.hourly.time[i]
        assert reading.temperature_c == response.hourly.temperature_2m[i]
        assert reading.precipitation_mm == response.hourly.precipitation[i]


def test_identical_readings_are_equal():
    a = HourlyReading("2026-04-08T00:00", 10.0, 0.0)
    b = HourlyReading("2026-04-08T00:00", 10.0, 0.0)
    assert a == b