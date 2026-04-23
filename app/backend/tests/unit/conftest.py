"""Fixtures for tests under this directory only.

Parent ``tests/conftest.py`` (autouse) runs ``init_database()`` once per session
and TRUNCATE before every test so integration tests can use PostgreSQL. Unit
tests here only exercise pure logic and must not require a live DB. Redefining
``db_setup`` and ``clean_db`` with the same names overrides the parent fixtures
for ``tests/unit/`` (pytest: nearer conftest wins), so we skip DB setup and
TRUNCATE while keeping the same test layout as the rest of ``tests/``.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def db_setup():
    """Override parent: do not run ``init_database()`` (no Postgres for unit tests)."""
    yield


@pytest.fixture(autouse=True)
def clean_db():
    """Override parent: do not TRUNCATE (no app_config usage in these tests)."""
    yield
