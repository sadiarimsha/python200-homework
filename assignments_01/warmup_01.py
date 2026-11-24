# --- Classes ---

# Q1 and Q2: Thermometer
class Thermometer:
    def __init__(self, location, readings=None):
        self.location = location
        # None-then-create avoids sharing one list between all objects
        self.readings = readings if readings is not None else []

    def add(self, reading):
        self.readings.append(reading)

    def average(self):
        if not self.readings:
            return None
        return sum(self.readings) / len(self.readings)

    def hottest(self):
        if not self.readings:
            return None
        return max(self.readings)

    def __repr__(self):
        avg = self.average()
        if avg is not None:
            avg = round(avg, 1)
        return (
            f"Thermometer(location={self.location!r}, "
            f"n_readings={len(self.readings)}, average={avg})"
        )


charlotte = Thermometer("Charlotte")
for temp in [14.2, 18.5, 21.0, 20.7]:
    charlotte.add(temp)

print(charlotte.average())
print(charlotte.hottest())
# Q1: If there are no readings, sum([]) / len([]) divides by zero and the
# program crashes with ZeroDivisionError. The check returns None instead,
# so an empty thermometer doesn't break the code.

print(charlotte)
print([charlotte, Thermometer("Chicago", [5.0, 7.5])])
# Q2: Without __repr__, Python shows something like
# <__main__.Thermometer object at 0x...>. That only tells me the type and a
# memory address, so when I debug a list of objects I can't see any values.


# Q3: TemperatureAlert
class TemperatureAlert:
    def __init__(self, threshold=30.0):
        self.threshold = threshold

    def breaches(self, thermometer):
        return [r for r in thermometer.readings if r > self.threshold]


mild_alert = TemperatureAlert(threshold=20.0)
strict_alert = TemperatureAlert(threshold=15.0)
print(mild_alert.breaches(charlotte))
print(strict_alert.breaches(charlotte))
# Q3: The threshold is stored on the alert so I set it once and reuse it.
# With twenty thermometers I can call breaches() on each one without typing
# the threshold every time, and the rule lives in one place.


# --- Dataclasses ---
from dataclasses import dataclass, field, FrozenInstanceError


# Q1 and Q2: Station (written as a dataclass, then made frozen)
@dataclass(frozen=True)
class Station:
    """A weather station with its identity, location, and elevation (meters)."""
    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float


station_a = Station("KCLT", "Charlotte", 35.21, -80.94, 228.0)
station_b = Station("KCLT", "Charlotte", 35.21, -80.94, 228.0)
print(station_a == station_b)
# Q1: This is True because a dataclass writes __eq__ for me, and it compares
# the field values. The original hand-written class would print False,
# because by default Python compares whether they are the same object, and
# these are two separate objects.

# Q2
try:
    station_a.elevation = 300.0
except FrozenInstanceError as e:
    print(e)

station_c = Station("KORD", "Chicago", 41.98, -87.90, 201.0)
print(len({station_a, station_b, station_c}))  # expect 2
# Q2: frozen=True also makes the objects hashable, so they can go in a set
# or be dictionary keys. That is why the two identical stations count as one
# and the set has length 2. It's useful here because a station's details
# shouldn't change after it's created.


# Q3: StationBatch
# First attempt (commented out because it crashes the file):
# @dataclass
# class BrokenBatch:
#     region: str
#     stations: list[Station] = []
#
# Error:
# ValueError: mutable default <class 'list'> for field stations is not
# allowed: use default_factory
#
# Python refuses because a plain [] is created once, when the class is
# defined, so every StationBatch would share the same list. Adding a station
# to one batch would add it to all of them. default_factory=list builds a
# new empty list for each object.

@dataclass
class StationBatch:
    """A group of stations in one region."""
    region: str
    stations: list[Station] = field(default_factory=list)

    def add(self, station: Station) -> None:
        """Add a station to the batch."""
        self.stations.append(station)

    def highest(self) -> Station | None:
        """Return the station with the greatest elevation, or None if empty."""
        if not self.stations:
            return None
        return max(self.stations, key=lambda s: s.elevation)


batch = StationBatch("Midwest")
print(batch.highest())  # None
batch.add(station_a)
batch.add(station_c)
print(batch.highest())


# --- Pydantic ---
from pydantic import BaseModel, Field, ValidationError, model_validator


class Reading(BaseModel):
    """One weather observation from a station."""
    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)

    # Q4: cross-field rule
    @model_validator(mode="after")
    def check_sensor(self):
        if self.humidity == 0.0 and self.temperature_c < -40:
            raise ValueError("humidity 0.0 with temperature below -40 suggests a failed sensor")
        return self


# Q1
good = Reading(station_id="KCLT", timestamp="2026-04-08T12:00", temperature_c=18.5, humidity=55)
print(good)

# Q2: three separate failures
for bad in [
    {"station_id": "KCLT", "temperature_c": 18.5, "humidity": 55},  # missing timestamp
    {"station_id": "KCLT", "timestamp": "t", "temperature_c": 150.0, "humidity": 55},
    {"station_id": "KCLT", "timestamp": "t", "temperature_c": 18.5, "humidity": "very humid"},
]:
    try:
        Reading(**bad)
    except ValidationError as e:
        print(e)

coerced = Reading(station_id="KCLT", timestamp="t", temperature_c="21.5", humidity=40)
print(coerced)
print(type(coerced.temperature_c), type(coerced.humidity))
# Q2: Pydantic accepts "21.5" because it can be safely converted to the
# number 21.5. It rejects "very humid" because there is no number in it to
# convert. If the value can be read as the right type, it converts it; if
# not, it raises an error.

# Q3: several errors at once
try:
    Reading(station_id="ab", temperature_c="hot", humidity=50)
except ValidationError as e:
    for err in e.errors():
        print(err["loc"], err["msg"])
# Q3: There were 3 errors (station_id too short, timestamp missing,
# temperature_c not a number). Seeing all of them at once is better because
# I can fix everything in one pass instead of fixing one, rerunning, and
# finding the next.

# Q4
print(Reading(station_id="KCLT", timestamp="t", temperature_c=-45, humidity=20))  # valid
try:
    Reading(station_id="KCLT", timestamp="t", temperature_c=-45, humidity=0.0)
except ValidationError as e:
    print(e)
# Q4: A Field constraint only looks at one field at a time. This rule needs
# to compare humidity and temperature together, so it needs a model
# validator that sees the whole object.


# --- pytest ---
import pytest


# Q1
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a temperature from Celsius to Fahrenheit."""
    return celsius * 9 / 5 + 32


def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(36.6) == pytest.approx(97.88)
# Q1: approx was necessary because decimal math in Python can be off by a
# tiny amount, so the result isn't exactly 97.88 and a plain == fails.
# approx allows a very small difference.


# Q2
def mean(values: list[float]) -> float:
    """Return the arithmetic mean of values. Raises ValueError if empty."""
    if not values:
        raise ValueError("cannot compute mean of an empty list")
    return sum(values) / len(values)


def test_mean_of_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        mean([])
# Q2: pytest.raises(ValueError) alone would pass for any ValueError, even
# one raised for a different reason. match= also checks the message, so I
# know it's the empty-list error I expected.


# Q3
@pytest.mark.parametrize(
    "values, expected",
    [
        ([1, 2, 3], 2),
        ([5], 5),
        ([-2, -4], -3),
        ([-1, 1, 3], 1),
    ],
)
def test_mean_values(values, expected):
    assert mean(values) == expected
# Q3 result: 6 passed in 0.12s
# Q3: One test function means the logic lives in one place, so changing it
# doesn't mean editing four copies. Each case still shows up as its own
# line in the output, so I can tell which input failed, and adding a new
# case is a single line.


# Q4: failure output after changing 9 / 5 to 9 / 4
#     def test_celsius_to_fahrenheit():
#         assert celsius_to_fahrenheit(0) == 32
# >       assert celsius_to_fahrenheit(100) == 212
# E       assert 257.0 == 212
# E        +  where 257.0 = celsius_to_fahrenheit(100)
#
# warmup_01.py:194: AssertionError
#
# Fixed: changed 9 / 4 back to 9 / 5, and all tests pass again.
#
# Answer: pytest showed the actual value (257.0), the expected value (212),
# and the exact line and call that failed. A bare "assertion failed" would
# make me hunt for which check broke and what the function returned.