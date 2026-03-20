"""Shared fixtures for SQL tool tests.

Requires a running PostgreSQL instance (podman-compose or CI service container)
with the env vars DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD set.
"""

import pytest

from database import get_connection, init_database


@pytest.fixture(scope="session", autouse=True)
def db_setup():
    """Create tables once for the entire test session."""
    init_database()


@pytest.fixture(autouse=True)
def clean_db():
    """Truncate all data before every test (respecting FK order)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("TRUNCATE detection_observations, detection_tracks CASCADE")
        conn.commit()
