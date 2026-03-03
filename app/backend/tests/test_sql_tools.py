"""Deep tests for the 7 PPE SQL LangChain tools + internal helpers.

Each test class seeds known data, invokes the tool, and asserts exact results.
"""

import json
from datetime import datetime, date, timedelta

import pytest

from seed import (
    seed_data,
    NOW,
    YESTERDAY,
    TWO_DAYS_AGO,
    FAR_FUTURE,
    # FIVE_MIN_AGO,
    # TWO_HOURS_AGO,
    # FAR_PAST,
)
from tools.sql import (
    count_ppe_violations,
    get_compliance_rate,
    list_violators,
    get_person_ppe_timeline,
    get_daily_compliance_summary,
    count_persons_seen,
    get_active_persons,
    _validate_ppe_type,
    _parse_time,
    _ts,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestHelpers:
    """Tests for internal helper functions."""

    @pytest.mark.parametrize("ppe_type", ["hardhat", "vest", "mask"])
    def test_validate_ppe_type_accepts_valid(self, ppe_type):
        assert _validate_ppe_type(ppe_type) == ppe_type

    def test_validate_ppe_type_normalises_case_and_whitespace(self):
        assert _validate_ppe_type("  Hardhat  ") == "hardhat"
        assert _validate_ppe_type("VEST") == "vest"

    @pytest.mark.parametrize("bad", ["helmet", "", "DROP TABLE", "safety vest", "123"])
    def test_validate_ppe_type_rejects_invalid(self, bad):
        with pytest.raises(ValueError, match="Invalid ppe_type"):
            _validate_ppe_type(bad)

    def test_parse_time_returns_default_when_none(self):
        default = datetime(2025, 1, 1, 12, 0, 0)
        assert _parse_time(None, default) == default

    def test_parse_time_parses_iso_string(self):
        result = _parse_time("2025-06-15T14:30:00", datetime.min)
        assert result == datetime(2025, 6, 15, 14, 30, 0)

    def test_ts_formats_datetime(self):
        dt = datetime(2025, 3, 15, 9, 5, 30)
        assert _ts(dt) == "2025-03-15 09:05:30"

    def test_ts_formats_date(self):
        d = date(2025, 12, 1)
        assert _ts(d) == "2025-12-01"


# ═══════════════════════════════════════════════════════════════════════════
# Tool: count_ppe_violations
# ═══════════════════════════════════════════════════════════════════════════


class TestCountPpeViolations:
    def test_hardhat_violations_returns_exact_count(self):
        seed_data()
        # Persons 2 and 3 both have hardhat=False observations
        wide_start = (TWO_DAYS_AGO - timedelta(days=1)).isoformat()
        wide_end = (NOW + timedelta(hours=1)).isoformat()
        result = json.loads(
            count_ppe_violations.invoke(
                {
                    "ppe_type": "hardhat",
                    "start_time": wide_start,
                    "end_time": wide_end,
                }
            )
        )
        assert result["unique_persons_without_ppe"] == 2
        assert result["ppe_type"] == "hardhat"

    def test_vest_violations_returns_one_person(self):
        seed_data()
        # Only person 3 has vest=False
        wide_start = (TWO_DAYS_AGO - timedelta(days=1)).isoformat()
        wide_end = (NOW + timedelta(hours=1)).isoformat()
        result = json.loads(
            count_ppe_violations.invoke(
                {
                    "ppe_type": "vest",
                    "start_time": wide_start,
                    "end_time": wide_end,
                }
            )
        )
        assert result["unique_persons_without_ppe"] == 1

    def test_time_range_excluding_all_data_returns_zero(self):
        seed_data()
        result = json.loads(
            count_ppe_violations.invoke(
                {
                    "ppe_type": "hardhat",
                    "start_time": FAR_FUTURE.isoformat(),
                    "end_time": (FAR_FUTURE + timedelta(hours=1)).isoformat(),
                }
            )
        )
        assert result["unique_persons_without_ppe"] == 0

    def test_default_time_range_finds_recent_data(self):
        seed_data()
        # Person 2 has hardhat=False observations within the last 24h
        result = json.loads(count_ppe_violations.invoke({"ppe_type": "hardhat"}))
        assert result["unique_persons_without_ppe"] >= 1

    def test_invalid_ppe_type_raises_error(self):
        with pytest.raises(Exception):
            count_ppe_violations.invoke({"ppe_type": "helmet"})


# ═══════════════════════════════════════════════════════════════════════════
# Tool: get_compliance_rate
# ═══════════════════════════════════════════════════════════════════════════


class TestGetComplianceRate:
    def test_mixed_compliance_returns_correct_percentage(self):
        seed_data()
        wide_start = (TWO_DAYS_AGO - timedelta(days=1)).isoformat()
        wide_end = (NOW + timedelta(hours=1)).isoformat()
        result = json.loads(
            get_compliance_rate.invoke(
                {
                    "ppe_type": "hardhat",
                    "start_time": wide_start,
                    "end_time": wide_end,
                }
            )
        )
        # Person 1: 3x True, Person 2: 3x False, Person 3: 2x False
        # compliant=3, non_compliant=5, total=8
        assert result["compliant"] == 3
        assert result["non_compliant"] == 5
        assert result["total_observations"] == 8
        assert result["compliance_rate_percent"] == 37.5

    def test_fully_compliant_ppe_type_returns_100(self):
        seed_data()
        wide_start = (TWO_DAYS_AGO - timedelta(days=1)).isoformat()
        wide_end = (NOW + timedelta(hours=1)).isoformat()
        # mask: person 1 has 3x True, person 3 has 2x True, person 2 has 3x None
        # Only non-NULL are counted: compliant=5, non_compliant=0, total=5
        result = json.loads(
            get_compliance_rate.invoke(
                {
                    "ppe_type": "mask",
                    "start_time": wide_start,
                    "end_time": wide_end,
                }
            )
        )
        assert result["compliance_rate_percent"] == 100.0
        assert result["non_compliant"] == 0

    def test_empty_time_range_returns_zero(self):
        seed_data()
        result = json.loads(
            get_compliance_rate.invoke(
                {
                    "ppe_type": "hardhat",
                    "start_time": FAR_FUTURE.isoformat(),
                    "end_time": (FAR_FUTURE + timedelta(hours=1)).isoformat(),
                }
            )
        )
        assert result["total_observations"] == 0
        assert result["compliance_rate_percent"] == 0.0

    def test_all_null_observations_returns_zero_total(self):
        seed_data()
        # Person 2 has mask=None at NOW-15min and NOW-8min.
        # Person 1 has mask=True at NOW-10min which overlaps.
        # Use a narrow window that only hits person 2's NOW-15min observation.
        start = (NOW - timedelta(minutes=16)).isoformat()
        end_t = (NOW - timedelta(minutes=14)).isoformat()
        result = json.loads(
            get_compliance_rate.invoke(
                {
                    "ppe_type": "mask",
                    "start_time": start,
                    "end_time": end_t,
                }
            )
        )
        assert result["total_observations"] == 0
        assert result["compliance_rate_percent"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Tool: list_violators
# ═══════════════════════════════════════════════════════════════════════════


class TestListViolators:
    def test_returns_violators_ordered_by_count(self):
        seed_data()
        wide_start = (TWO_DAYS_AGO - timedelta(days=1)).isoformat()
        wide_end = (NOW + timedelta(hours=1)).isoformat()
        rows = json.loads(
            list_violators.invoke(
                {
                    "ppe_type": "hardhat",
                    "start_time": wide_start,
                    "end_time": wide_end,
                }
            )
        )
        assert len(rows) == 2
        # Person 2 has 3 hardhat violations, person 3 has 2
        assert rows[0]["track_id"] == 2
        assert rows[0]["violation_count"] == 3
        assert rows[1]["track_id"] == 3
        assert rows[1]["violation_count"] == 2

    def test_respects_limit(self):
        seed_data()
        wide_start = (TWO_DAYS_AGO - timedelta(days=1)).isoformat()
        wide_end = (NOW + timedelta(hours=1)).isoformat()
        rows = json.loads(
            list_violators.invoke(
                {
                    "ppe_type": "hardhat",
                    "start_time": wide_start,
                    "end_time": wide_end,
                    "limit": 1,
                }
            )
        )
        assert len(rows) == 1
        assert rows[0]["track_id"] == 2

    def test_includes_first_and_last_violation_timestamps(self):
        seed_data()
        wide_start = (TWO_DAYS_AGO - timedelta(days=1)).isoformat()
        wide_end = (NOW + timedelta(hours=1)).isoformat()
        rows = json.loads(
            list_violators.invoke(
                {
                    "ppe_type": "hardhat",
                    "start_time": wide_start,
                    "end_time": wide_end,
                }
            )
        )
        for row in rows:
            assert "first_violation" in row
            assert "last_violation" in row
            first = datetime.fromisoformat(row["first_violation"])
            last = datetime.fromisoformat(row["last_violation"])
            assert first <= last

    def test_empty_range_returns_empty_list(self):
        seed_data()
        rows = json.loads(
            list_violators.invoke(
                {
                    "ppe_type": "hardhat",
                    "start_time": FAR_FUTURE.isoformat(),
                    "end_time": (FAR_FUTURE + timedelta(hours=1)).isoformat(),
                }
            )
        )
        assert rows == []


# ═══════════════════════════════════════════════════════════════════════════
# Tool: get_person_ppe_timeline
# ═══════════════════════════════════════════════════════════════════════════


class TestGetPersonPpeTimeline:
    def test_returns_correct_observations_for_person(self):
        seed_data()
        rows = json.loads(get_person_ppe_timeline.invoke({"track_id": 2}))
        assert len(rows) == 3
        for row in rows:
            assert row["hardhat"] is False
            assert row["vest"] is True
            assert row["mask"] is None

    def test_results_ordered_most_recent_first(self):
        seed_data()
        rows = json.loads(get_person_ppe_timeline.invoke({"track_id": 1}))
        timestamps = [datetime.fromisoformat(r["timestamp"]) for r in rows]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_respects_limit(self):
        seed_data()
        rows = json.loads(
            get_person_ppe_timeline.invoke(
                {
                    "track_id": 1,
                    "limit": 1,
                }
            )
        )
        assert len(rows) == 1

    def test_nonexistent_track_id_returns_empty(self):
        seed_data()
        rows = json.loads(get_person_ppe_timeline.invoke({"track_id": 9999}))
        assert rows == []


# ═══════════════════════════════════════════════════════════════════════════
# Tool: get_daily_compliance_summary
# ═══════════════════════════════════════════════════════════════════════════


class TestGetDailyComplianceSummary:
    def test_returns_per_day_rows(self):
        seed_data()
        start = (TWO_DAYS_AGO - timedelta(days=1)).strftime("%Y-%m-%d")
        end = (NOW + timedelta(days=1)).strftime("%Y-%m-%d")
        rows = json.loads(
            get_daily_compliance_summary.invoke(
                {
                    "ppe_type": "hardhat",
                    "start_date": start,
                    "end_date": end,
                }
            )
        )
        # Data spans up to 3 distinct days (two_days_ago, yesterday, today)
        assert len(rows) >= 2
        for row in rows:
            assert "day" in row
            assert "total_observations" in row
            assert "violations" in row
            assert "compliance_rate_percent" in row
            assert row["total_observations"] > 0

    def test_single_day_range(self):
        seed_data()
        day_str = YESTERDAY.strftime("%Y-%m-%d")
        rows = json.loads(
            get_daily_compliance_summary.invoke(
                {
                    "ppe_type": "hardhat",
                    "start_date": day_str,
                    "end_date": day_str,
                }
            )
        )
        assert len(rows) == 1
        assert rows[0]["day"] == day_str

    def test_empty_range_returns_empty_list(self):
        seed_data()
        rows = json.loads(
            get_daily_compliance_summary.invoke(
                {
                    "ppe_type": "hardhat",
                    "start_date": FAR_FUTURE.strftime("%Y-%m-%d"),
                    "end_date": (FAR_FUTURE + timedelta(days=1)).strftime("%Y-%m-%d"),
                }
            )
        )
        assert rows == []


# ═══════════════════════════════════════════════════════════════════════════
# Tool: count_persons_seen
# ═══════════════════════════════════════════════════════════════════════════


class TestCountPersonsSeen:
    def test_correct_count_with_known_data(self):
        seed_data()
        wide_start = (TWO_DAYS_AGO - timedelta(days=1)).isoformat()
        wide_end = (NOW + timedelta(hours=1)).isoformat()
        result = json.loads(
            count_persons_seen.invoke(
                {
                    "start_time": wide_start,
                    "end_time": wide_end,
                }
            )
        )
        assert result["unique_persons"] == 3

    def test_time_range_excludes_old_records(self):
        seed_data()
        # Only observations from today (persons 1 and 2 have today observations)
        start = NOW.replace(hour=0, minute=0, second=0).isoformat()
        end = (NOW + timedelta(hours=1)).isoformat()
        result = json.loads(
            count_persons_seen.invoke(
                {
                    "start_time": start,
                    "end_time": end,
                }
            )
        )
        assert result["unique_persons"] == 2

    def test_empty_range_returns_zero(self):
        seed_data()
        result = json.loads(
            count_persons_seen.invoke(
                {
                    "start_time": FAR_FUTURE.isoformat(),
                    "end_time": (FAR_FUTURE + timedelta(hours=1)).isoformat(),
                }
            )
        )
        assert result["unique_persons"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# Tool: get_active_persons
# ═══════════════════════════════════════════════════════════════════════════


class TestGetActivePersons:
    def test_default_finds_active_persons(self):
        seed_data()
        # Persons 1 and 2 have last_seen = 5 min ago (within 10-min window)
        rows = json.loads(get_active_persons.invoke({}))
        track_ids = {r["track_id"] for r in rows}
        assert 1 in track_ids
        assert 2 in track_ids
        # Person 3 last_seen 2 hours ago, should not be in default 10-min window
        assert 3 not in track_ids

    def test_small_window_may_find_fewer(self):
        seed_data()
        # 1 minute ago — persons 1 and 2 were last seen 5 min ago, so excluded
        rows = json.loads(get_active_persons.invoke({"minutes_ago": 1}))
        track_ids = {r["track_id"] for r in rows}
        assert 3 not in track_ids

    def test_large_window_finds_all(self):
        seed_data()
        # 180 minutes = 3 hours; person 3 was last seen 2 hours ago
        rows = json.loads(get_active_persons.invoke({"minutes_ago": 180}))
        track_ids = {r["track_id"] for r in rows}
        assert track_ids == {1, 2, 3}
