import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from weatherkit import WeatherResponse

# A plain relative path like "weather_raw.json" is resolved from the folder
# where pytest is run, not from where this test file lives. If someone runs
# pytest from another folder, Python looks in the wrong place and can't find
# the file. Path(__file__) starts from this file's own location, so it works
# from anywhere.
JSON_PATH = Path(__file__).parent.parent / "weather_raw.json"


def make_raw(time, temps, precip, latitude=35.2):
    """Build a small hand-written API response for tests."""
    return {
        "latitude": latitude,
        "longitude": -80.8,
        "timezone": "America/New_York",
        "elevation": 230.0,
        "hourly": {
            "time": time,
            "temperature_2m": temps,
            "precipitation": precip,
        },
    }


def test_real_file_validates():
    with open(JSON_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    response = WeatherResponse.model_validate(raw)
    assert len(response.hourly.time) == 168


def test_bad_latitude_raises():
    raw = make_raw(["a", "b"], [1.0, 2.0], [0.0, 0.0], latitude=200.0)
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(raw)


def test_mismatched_lengths_raise():
    raw = make_raw(["a", "b", "c"], [1.0, 2.0], [0.0, 0.0, 0.0])  # temps one short
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(raw)


def test_null_temperature_raises():
    raw = make_raw(["a", "b", "c"], [1.0, None, 3.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(raw)