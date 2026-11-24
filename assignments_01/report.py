import json
from pathlib import Path

from weatherkit import WeatherResponse, to_readings, DailyAggregator


def load_response() -> WeatherResponse:
    """Load weather_raw.json and validate it."""
    path = Path(__file__).parent / "weather_raw.json"
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return WeatherResponse.model_validate(raw)


def main() -> None:
    response = load_response()
    print(f"Latitude: {response.latitude}")
    print(f"Timezone: {response.timezone}")
    print(f"Hourly observations: {len(response.hourly.time)}")  # expect 168
    print()

    readings = to_readings(response)
    aggregator = DailyAggregator()  # min_hours defaults to 24
    summaries = aggregator.summarize(readings)

    print(f"{'Date':<12}{'High':>7}{'Low':>7}{'Precip':>9}{'Range':>8}")
    for s in summaries:
        print(
            f"{s.date:<12}{s.temp_max:>7.1f}{s.temp_min:>7.1f}"
            f"{s.precipitation_sum:>9.1f}{s.temp_range():>8.1f}"
        )

    dropped = aggregator.incomplete_days(readings)
    if dropped:
        print(f"\nWARNING: dropped incomplete days: {', '.join(dropped)}")


# Task 6: Without the guard, main() would run as soon as someone imports
# report.py, even if they only wanted one helper function. It would print
# the whole table and load the file unexpectedly. The guard makes main()
# run only when I run the file directly.
if __name__ == "__main__":
    main()


# Task 7 reflection
# 1. Rejecting the whole file on one null is right when accuracy matters,
#    like a safety alert, because one bad value could make a wrong answer
#    look correct. I would tolerate the gap for a daily summary, where one
#    missing hour out of 24 barely changes the result. I would change the
#    schema to list[float | None] and skip the None values when computing.
#
# 2. If the pipeline runs at noon, today only has about 12 hours, so the
#    high and low would be wrong because the afternoon hasn't happened yet.
#    The min_hours rule drops that day, and incomplete_days() tells me which
#    date was dropped, so it doesn't disappear silently.
#
# 3. In Week 10, I can import the code from any script with
#    from weatherkit import ..., instead of copying and pasting it.