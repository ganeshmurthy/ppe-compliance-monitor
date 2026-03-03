"""LangChain tools for read-only PPE compliance database queries.

Every tool uses a read-only PostgreSQL connection so there is zero risk of
accidental data modification. Column names are validated against a whitelist.
"""

import json
from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import Optional

from langchain_core.tools import tool
from psycopg2.extras import RealDictCursor

from database import get_readonly_connection
from logger import get_logger

log = get_logger(__name__)

_VALID_PPE_TYPES = {"hardhat", "vest", "mask"}


def _validate_ppe_type(ppe_type: str) -> str:
    ppe_type = ppe_type.strip().lower()
    if ppe_type not in _VALID_PPE_TYPES:
        raise ValueError(
            f"Invalid ppe_type '{ppe_type}'. Must be one of: {', '.join(sorted(_VALID_PPE_TYPES))}"
        )
    return ppe_type


def _parse_time(value: Optional[str], default: datetime) -> datetime:
    if value is None:
        return default
    return datetime.fromisoformat(value)


def _ts(dt: datetime | date) -> str:
    """Format a datetime/date as a PostgreSQL timestamp literal."""
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return dt.strftime("%Y-%m-%d")


def _serialize(rows: list[dict]) -> str:
    """JSON-serialize query results, converting datetimes to ISO strings."""

    def _default(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    return json.dumps(rows, default=_default)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def count_ppe_violations(
    ppe_type: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> str:
    """Count unique people who were NOT wearing a specific PPE item in a time range.

    Args:
        ppe_type: Which PPE to check — "hardhat", "vest", or "mask".
        start_time: ISO-format start (e.g. "2025-06-01T00:00:00"). Defaults to 24 h ago.
        end_time: ISO-format end. Defaults to now.
    """
    col = _validate_ppe_type(ppe_type)
    now = datetime.now()
    start = _parse_time(start_time, now - timedelta(hours=24))
    end = _parse_time(end_time, now)

    with get_readonly_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT COUNT(DISTINCT track_id) AS violation_count "
            f"FROM person_observations "
            f"WHERE {col} = FALSE AND timestamp BETWEEN '{_ts(start)}' AND '{_ts(end)}'"
        )
        count = cur.fetchone()[0]

    return json.dumps(
        {
            "ppe_type": col,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "unique_persons_without_ppe": count,
        }
    )


@tool
def get_compliance_rate(
    ppe_type: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> str:
    """Get the compliance percentage for a PPE type in a time range.

    Args:
        ppe_type: Which PPE to check — "hardhat", "vest", or "mask".
        start_time: ISO-format start. Defaults to 24 h ago.
        end_time: ISO-format end. Defaults to now.
    """
    col = _validate_ppe_type(ppe_type)
    now = datetime.now()
    start = _parse_time(start_time, now - timedelta(hours=24))
    end = _parse_time(end_time, now)

    with get_readonly_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT "
            f"  COUNT(*) FILTER (WHERE {col} = TRUE)  AS compliant, "
            f"  COUNT(*) FILTER (WHERE {col} = FALSE) AS non_compliant, "
            f"  COUNT(*) FILTER (WHERE {col} IS NOT NULL) AS total "
            f"FROM person_observations "
            f"WHERE timestamp BETWEEN '{_ts(start)}' AND '{_ts(end)}'"
        )
        row = cur.fetchone()
        compliant, non_compliant, total = row

    rate = round(compliant / total * 100, 2) if total > 0 else 0.0
    return json.dumps(
        {
            "ppe_type": col,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "compliant": compliant,
            "non_compliant": non_compliant,
            "total_observations": total,
            "compliance_rate_percent": rate,
        }
    )


@tool
def list_violators(
    ppe_type: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 20,
) -> str:
    """List person track IDs that violated a PPE rule, ordered by violation count.

    Args:
        ppe_type: Which PPE to check — "hardhat", "vest", or "mask".
        start_time: ISO-format start. Defaults to 24 h ago.
        end_time: ISO-format end. Defaults to now.
        limit: Max number of persons to return (default 20).
    """
    col = _validate_ppe_type(ppe_type)
    now = datetime.now()
    start = _parse_time(start_time, now - timedelta(hours=24))
    end = _parse_time(end_time, now)

    with get_readonly_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            f"SELECT "
            f"  track_id, "
            f"  COUNT(*) AS violation_count, "
            f"  MIN(timestamp) AS first_violation, "
            f"  MAX(timestamp) AS last_violation "
            f"FROM person_observations "
            f"WHERE {col} = FALSE AND timestamp BETWEEN '{_ts(start)}' AND '{_ts(end)}' "
            f"GROUP BY track_id "
            f"ORDER BY violation_count DESC "
            f"LIMIT {int(limit)}"
        )
        rows = [dict(r) for r in cur.fetchall()]

    return _serialize(rows)


@tool
def get_person_ppe_timeline(
    track_id: int,
    limit: int = 50,
) -> str:
    """Get the PPE observation history for a specific tracked person.

    Args:
        track_id: The person's unique track ID.
        limit: Max observations to return (default 50), most recent first.
    """
    with get_readonly_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            f"SELECT timestamp, hardhat, vest, mask "
            f"FROM person_observations "
            f"WHERE track_id = {int(track_id)} "
            f"ORDER BY timestamp DESC "
            f"LIMIT {int(limit)}"
        )
        rows = [dict(r) for r in cur.fetchall()]

    return _serialize(rows)


@tool
def get_daily_compliance_summary(
    ppe_type: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """Get per-day compliance breakdown over a date range.

    Args:
        ppe_type: Which PPE to check — "hardhat", "vest", or "mask".
        start_date: ISO date string (e.g. "2025-06-01"). Defaults to 7 days ago.
        end_date: ISO date string. Defaults to today.
    """
    col = _validate_ppe_type(ppe_type)
    today = datetime.now().date()
    start = (
        datetime.fromisoformat(start_date).date()
        if start_date
        else today - timedelta(days=7)
    )
    end = datetime.fromisoformat(end_date).date() if end_date else today

    with get_readonly_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            f"SELECT "
            f"  DATE_TRUNC('day', timestamp)::date AS day, "
            f"  COUNT(*) FILTER (WHERE {col} IS NOT NULL) AS total_observations, "
            f"  COUNT(*) FILTER (WHERE {col} = FALSE) AS violations, "
            f"  CASE "
            f"    WHEN COUNT(*) FILTER (WHERE {col} IS NOT NULL) > 0 THEN "
            f"      ROUND(COUNT(*) FILTER (WHERE {col} = TRUE)::numeric "
            f"        / COUNT(*) FILTER (WHERE {col} IS NOT NULL) * 100, 2) "
            f"    ELSE 0 "
            f"  END AS compliance_rate_percent "
            f"FROM person_observations "
            f"WHERE timestamp::date BETWEEN '{_ts(start)}' AND '{_ts(end)}' "
            f"GROUP BY day "
            f"ORDER BY day"
        )
        rows = [dict(r) for r in cur.fetchall()]

    return _serialize(rows)


@tool
def count_persons_seen(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> str:
    """Count total unique persons detected in a time range.

    Args:
        start_time: ISO-format start. Defaults to 24 h ago.
        end_time: ISO-format end. Defaults to now.
    """
    now = datetime.now()
    start = _parse_time(start_time, now - timedelta(hours=24))
    end = _parse_time(end_time, now)

    with get_readonly_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT COUNT(DISTINCT track_id) AS person_count "
            f"FROM person_observations "
            f"WHERE timestamp BETWEEN '{_ts(start)}' AND '{_ts(end)}'"
        )
        count = cur.fetchone()[0]

    return json.dumps(
        {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "unique_persons": count,
        }
    )


@tool
def get_active_persons(minutes_ago: int = 10) -> str:
    """List persons currently on-site (seen within the last N minutes).

    Args:
        minutes_ago: Look-back window in minutes (default 10).
    """
    cutoff = datetime.now() - timedelta(minutes=int(minutes_ago))

    with get_readonly_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            f"SELECT track_id, first_seen, last_seen "
            f"FROM persons "
            f"WHERE last_seen >= '{_ts(cutoff)}' "
            f"ORDER BY last_seen DESC"
        )
        rows = [dict(r) for r in cur.fetchall()]

    return _serialize(rows)
