"""Deleting app_config cascades to dependent tables."""

from datetime import datetime, timezone

from database import (
    delete_config,
    get_connection,
    insert_config,
    insert_detection_observation,
    insert_detection_track,
    replace_detection_classes,
)


def test_delete_config_cascades_tracks_and_observations():
    cid = insert_config("http://ovms:8080", "/tmp/v.mp4", "ppe")
    replace_detection_classes(cid, [(0, "Person", True, True)])
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM detection_classes WHERE app_config_id = %s",
            (cid,),
        )
        dc_id = cur.fetchone()[0]
    now = datetime.now(timezone.utc)
    insert_detection_track(1, dc_id, now, now)
    insert_detection_observation(1, now, {"hardhat": True})
    assert delete_config(cid) is True
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM app_config WHERE id = %s", (cid,))
        assert cur.fetchone()[0] == 0
        cur.execute(
            "SELECT COUNT(*) FROM detection_classes WHERE app_config_id = %s",
            (cid,),
        )
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT COUNT(*) FROM detection_tracks")
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT COUNT(*) FROM detection_observations")
        assert cur.fetchone()[0] == 0


def test_delete_config_missing_returns_false():
    assert delete_config(999_999) is False
