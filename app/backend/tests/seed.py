"""Shared test constants and seed data helper.

Separated from conftest.py so both conftest and test modules can import it.
"""

from datetime import datetime, timedelta
from database import insert_person, insert_observation

# ── Timestamps used across all seed data ─────────────────────────────────
NOW = datetime.now()
FIVE_MIN_AGO = NOW - timedelta(minutes=5)
TWO_HOURS_AGO = NOW - timedelta(hours=2)
YESTERDAY = NOW - timedelta(days=1)
TWO_DAYS_AGO = NOW - timedelta(days=2)
FAR_FUTURE = NOW + timedelta(days=365)
FAR_PAST = NOW - timedelta(days=365)


def seed_data():
    """Insert a controlled dataset of 3 persons with known PPE statuses.

    Person 1 (track_id=1): fully compliant, active (last_seen = 5 min ago)
      - 3 observations (yesterday x2, today x1): hardhat=T, vest=T, mask=T

    Person 2 (track_id=2): hardhat violator, active (last_seen = 5 min ago)
      - 3 observations (yesterday x1, today x2): hardhat=F, vest=T, mask=None

    Person 3 (track_id=3): hardhat + vest violator, inactive (last_seen = 2h ago)
      - 2 observations (two_days_ago x1, yesterday x1): hardhat=F, vest=F, mask=T
    """
    insert_person(1, YESTERDAY, FIVE_MIN_AGO)
    insert_observation(1, YESTERDAY, hardhat=True, vest=True, mask=True)
    insert_observation(
        1, YESTERDAY + timedelta(hours=1), hardhat=True, vest=True, mask=True
    )
    insert_observation(
        1, NOW - timedelta(minutes=10), hardhat=True, vest=True, mask=True
    )

    insert_person(2, YESTERDAY, FIVE_MIN_AGO)
    insert_observation(
        2, YESTERDAY + timedelta(hours=2), hardhat=False, vest=True, mask=None
    )
    insert_observation(
        2, NOW - timedelta(minutes=15), hardhat=False, vest=True, mask=None
    )
    insert_observation(
        2, NOW - timedelta(minutes=8), hardhat=False, vest=True, mask=None
    )

    insert_person(3, TWO_DAYS_AGO, TWO_HOURS_AGO)
    insert_observation(
        3, TWO_DAYS_AGO + timedelta(hours=3), hardhat=False, vest=False, mask=True
    )
    insert_observation(
        3, YESTERDAY + timedelta(hours=5), hardhat=False, vest=False, mask=True
    )
