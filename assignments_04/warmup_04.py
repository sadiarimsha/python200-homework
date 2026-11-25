import json
from dataclasses import dataclass

import pytest


# Q1
class ModelInfo:
    def __init__(self, path):
        with open(path) as f:
            self.data = json.load(f)

    def feature_count(self) -> int:
        return len(self.data["features"])


# loading in __init__ reads the file once, not on every call


# Q2
@dataclass
class DayForecast:
    date: str
    high_c: float
    will_rain: bool

    def summary(self) -> str:
        rain = "rain expected" if self.will_rain else "no rain expected"
        return f"{self.date}: high {self.high_c}C, {rain}"


# @dataclass wrote __init__ for me


# Q3
def parse_temperature(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"invalid temperature: {text}")


def test_parse_temperature_valid():
    assert parse_temperature("24.1") == pytest.approx(24.1)


def test_parse_temperature_bad_input_raises():
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")


# without match, any ValueError passes, even one with the wrong message

if __name__ == "__main__":
    with open("features.json", "w") as f:
        json.dump({"features": ["high", "low", "rain"]}, f)
    print(ModelInfo("features.json").feature_count())

    print(DayForecast("2023-06-15", 24.1, True).summary())
    print(DayForecast("2023-06-16", 27.0, False).summary())
