"""Shared test constants and seed data helper.

Separated from conftest.py so both conftest and test modules can import it.
"""

from datetime import datetime, timedelta
from database import (
    insert_config,
    replace_detection_classes,
    get_detection_class_by_name_and_config,
    get_all_configs,
    insert_detection_track,
    insert_detection_observation,
)

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
    person_class_id = _get_or_create_person_class_id()
    insert_detection_track(1, person_class_id, YESTERDAY, FIVE_MIN_AGO)
    insert_detection_observation(
        1, YESTERDAY, {"hardhat": True, "vest": True, "mask": True}
    )
    insert_detection_observation(
        1, YESTERDAY + timedelta(hours=1), {"hardhat": True, "vest": True, "mask": True}
    )
    insert_detection_observation(
        1, NOW - timedelta(minutes=10), {"hardhat": True, "vest": True, "mask": True}
    )

    insert_detection_track(2, person_class_id, YESTERDAY, FIVE_MIN_AGO)
    insert_detection_observation(
        2, YESTERDAY + timedelta(hours=2), {"hardhat": False, "vest": True}
    )
    insert_detection_observation(
        2, NOW - timedelta(minutes=15), {"hardhat": False, "vest": True}
    )
    insert_detection_observation(
        2, NOW - timedelta(minutes=8), {"hardhat": False, "vest": True}
    )

    insert_detection_track(3, person_class_id, TWO_DAYS_AGO, TWO_HOURS_AGO)
    insert_detection_observation(
        3,
        TWO_DAYS_AGO + timedelta(hours=3),
        {"hardhat": False, "vest": False, "mask": True},
    )
    insert_detection_observation(
        3,
        YESTERDAY + timedelta(hours=5),
        {"hardhat": False, "vest": False, "mask": True},
    )


def _get_or_create_person_class_id() -> int:
    """Ensure a config with Person class exists; return its detection_classes.id."""
    configs = get_all_configs()
    if not configs:
        config_id = insert_config("http://test", "rtsp://test", model_name="ppe")
        replace_detection_classes(
            config_id,
            [
                (0, "Person", True),
                (1, "Hardhat", False),
                (2, "NO-Hardhat", False),
                (3, "Safety Vest", False),
                (4, "NO-Safety Vest", False),
                (5, "Mask", False),
                (6, "NO-Mask", False),
            ],
        )
        configs = get_all_configs()
    config = configs[0]
    person = get_detection_class_by_name_and_config("Person", config["id"])
    if not person:
        replace_detection_classes(
            config["id"],
            [
                (0, "Person", True),
                (1, "Hardhat", False),
                (2, "NO-Hardhat", False),
                (3, "Safety Vest", False),
                (4, "NO-Safety Vest", False),
                (5, "Mask", False),
                (6, "NO-Mask", False),
            ],
        )
        person = get_detection_class_by_name_and_config("Person", config["id"])
    return person["id"]
