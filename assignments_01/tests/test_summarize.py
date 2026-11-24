
# Break test: I changed `<` to `<=` in DailyAggregator.summarize, so days
# with exactly 24 hours were dropped. 7 tests failed, including
# test_grouping_gives_two_summaries and test_max_and_min. I changed it back
# and all 16 tests pass again.

import pytest

from weatherkit import HourlyReading, DailyAggregator

def make_day(date, n_hours, base_temp=0.0, precip=0.1):
    """Build n_hours readings for one date. Temps rise by 1 each hour."""
    return [
        HourlyReading(f"{date}T{h:02d}:00", base_temp + h, precip)
        for h in range(n_hours)
    ]


@pytest.fixture
def two_full_days():
    return make_day("2026-04-08", 24) + make_day("2026-04-09", 24, base_temp=10.0)


@pytest.fixture
def full_plus_partial():
    # one full day (24 hours) and one partial day (only 6 hours)
    return make_day("2026-04-08", 24) + make_day("2026-04-09", 6)


def test_grouping_gives_two_summaries(two_full_days):
    summaries = DailyAggregator().summarize(two_full_days)
    assert len(summaries) == 2
    assert [s.date for s in summaries] == ["2026-04-08", "2026-04-09"]


def test_max_and_min(two_full_days):
    first = DailyAggregator().summarize(two_full_days)[0]
    assert first.temp_max == 23.0  # temps are 0..23
    assert first.temp_min == 0.0


def test_precipitation_sum(two_full_days):
    first = DailyAggregator().summarize(two_full_days)[0]
    assert first.precipitation_sum == pytest.approx(2.4)  # 24 x 0.1


def test_partial_day_dropped_and_reported(full_plus_partial):
    agg = DailyAggregator()
    summaries = agg.summarize(full_plus_partial)
    assert [s.date for s in summaries] == ["2026-04-08"]
    assert agg.incomplete_days(full_plus_partial) == ["2026-04-09"]


def test_lower_min_hours_keeps_partial_day(full_plus_partial):
    agg = DailyAggregator(min_hours=6)
    summaries = agg.summarize(full_plus_partial)
    assert len(summaries) == 2
    assert agg.incomplete_days(full_plus_partial) == []


@pytest.mark.parametrize(
    "min_hours, expected_days",
    [(24, 1), (7, 1), (6, 2), (1, 2)],
)
def test_min_hours_threshold(full_plus_partial, min_hours, expected_days):
    summaries = DailyAggregator(min_hours=min_hours).summarize(full_plus_partial)
    assert len(summaries) == expected_days