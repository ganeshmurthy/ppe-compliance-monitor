"""Load the pg_dump seed file into PostgreSQL for testing.

Usage:
    # As a module (from tests or scripts)
    from evals.load_seed import load_seed
    load_seed()

    # Standalone
    python -m evals.load_seed
"""

from pathlib import Path

from database import get_connection, init_database

SEED_SQL = Path(__file__).with_name("db_seed_data.sql")

SKIP_PREFIXES = ("--", "\\", "SET ", "SELECT pg_catalog.set_config")


def load_seed(sql_path: Path = SEED_SQL) -> dict:
    """Truncate tables, replay INSERT statements, and reset the sequence.

    Returns a summary dict with row counts.
    """
    init_database()

    statements: list[str] = []
    setval_stmt: str | None = None

    with open(sql_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or any(stripped.startswith(p) for p in SKIP_PREFIXES):
                if stripped.startswith("SELECT pg_catalog.setval"):
                    setval_stmt = stripped.rstrip(";") + ";"
                continue
            if stripped.startswith("INSERT INTO"):
                statements.append(stripped)

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("TRUNCATE person_observations, persons CASCADE")

        for stmt in statements:
            cur.execute(stmt)

        if setval_stmt:
            cur.execute(setval_stmt)

        cur.execute("SELECT count(*) FROM persons")
        persons_count = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM person_observations")
        observations_count = cur.fetchone()[0]

        conn.commit()

    return {"persons": persons_count, "person_observations": observations_count}
