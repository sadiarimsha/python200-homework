

from dataclasses import dataclass

from weatherkit.records import HourlyReading


@dataclass
class DailySummary:
    """A summary of one calendar day.

    Attributes:
        date: The day as "YYYY-MM-DD".
        temp_max: Highest temperature in Celsius.
        temp_min: Lowest temperature in Celsius.
        precipitation_sum: Total precipitation in millimeters.
        hours_observed: How many hourly readings were used.
    """
    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int

    def temp_range(self) -> float:
        """Return the difference between the high and the low."""
        return self.temp_max - self.temp_min


class DailyAggregator:
    """Groups hourly readings into daily summaries.

    Days with fewer than min_hours readings are dropped, because a daily
    maximum from a partial day can't be trusted.
    """

    def __init__(self, min_hours: int = 24):
        self.min_hours = min_hours

    def _group_by_date(
        self, readings: list[HourlyReading]
    ) -> dict[str, list[HourlyReading]]:
        """Put readings into a dict keyed by date (first 10 characters)."""
        groups: dict[str, list[HourlyReading]] = {}
        for reading in readings:
            date = reading.timestamp[:10]
            if date not in groups:
                groups[date] = []
            groups[date].append(reading)
        return groups

    def summarize(self, readings: list[HourlyReading]) -> list[DailySummary]:
        """Return one DailySummary per complete day, sorted by date."""
        summaries = []
        groups = self._group_by_date(readings)
        for date in sorted(groups):
            day = groups[date]
            if len(day) < self.min_hours:
                continue  # incomplete day: skip it
            temps = [r.temperature_c for r in day]
            summaries.append(
                DailySummary(
                    date=date,
                    temp_max=max(temps),
                    temp_min=min(temps),
                    precipitation_sum=sum(r.precipitation_mm for r in day),
                    hours_observed=len(day),
                )
            )
        return summaries

    def incomplete_days(self, readings: list[HourlyReading]) -> list[str]:
        """Return the dates that were dropped for having too few hours."""
        groups = self._group_by_date(readings)
        return [
            date for date in sorted(groups) if len(groups[date]) < self.min_hours
        ]