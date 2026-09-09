"""
agent/storage.py
─────────────────
Storage backends — where the collected call data gets saved.

Design: a simple abstract base class (StorageBackend) with two implementations:
  - SQLiteBackend  (default — no setup, file created automatically)
  - SupabaseBackend (Phase 4 upgrade — swap via DATABASE_URL in .env)

To switch from SQLite to Supabase, change one line in .env:
  DATABASE_URL=postgresql://...

Nothing else in the codebase needs to change.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.config import get_settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract interface
# ─────────────────────────────────────────────────────────────────────────────

class StorageBackend(ABC):
    """Every storage backend must implement these two methods."""

    @abstractmethod
    def save_call(self, result: dict[str, Any]) -> str:
        """
        Persist the result of one completed call.

        Args:
            result: The dict returned by ConversationManager.get_result()

        Returns:
            The call_id (for confirmation logging).
        """
        ...

    @abstractmethod
    def get_call(self, call_id: str) -> dict[str, Any] | None:
        """Retrieve a previously saved call by its ID."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# SQLite backend (default)
# ─────────────────────────────────────────────────────────────────────────────

class SQLiteBackend(StorageBackend):
    """
    Stores call results in a local SQLite database file.

    Schema (one table):
        agent_calls (
            call_id       TEXT PRIMARY KEY,
            started_at    TEXT,
            ended_at      TEXT,
            collected     TEXT,   ← JSON blob of the extracted field values
            missing_at_end TEXT,  ← JSON array of fields we couldn't fill
            turn_count    INTEGER,
            created_at    TEXT    ← when the row was inserted
        )
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            # Strip the "sqlite:///" prefix if present
            url = get_settings().database_url
            db_path = url.replace("sqlite:///", "")
        self._db_path = Path(db_path)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row   # lets us access columns by name
        return conn

    def _ensure_schema(self) -> None:
        """Create the table if it doesn't exist yet (idempotent)."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_calls (
                    call_id        TEXT PRIMARY KEY,
                    started_at     TEXT,
                    ended_at       TEXT,
                    collected      TEXT,
                    missing_at_end TEXT,
                    turn_count     INTEGER,
                    created_at     TEXT
                )
            """)
            conn.commit()
        logger.debug("SQLite schema verified at %s", self._db_path)

    def save_call(self, result: dict[str, Any]) -> str:
        call_id = result["call_id"]
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO agent_calls
                    (call_id, started_at, ended_at, collected, missing_at_end, turn_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    call_id,
                    result.get("started_at"),
                    result.get("ended_at"),
                    json.dumps(result.get("collected", {})),
                    json.dumps(result.get("missing_at_end", [])),
                    result.get("turn_count", 0),
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
        logger.info("Saved call %s to SQLite at %s", call_id, self._db_path)
        return call_id

    def get_call(self, call_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM agent_calls WHERE call_id = ?", (call_id,)
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["collected"] = json.loads(d["collected"])
        d["missing_at_end"] = json.loads(d["missing_at_end"])
        return d

    def list_calls(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent `limit` calls (for debugging)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_calls ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["collected"] = json.loads(d["collected"])
            d["missing_at_end"] = json.loads(d["missing_at_end"])
            result.append(d)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Supabase backend stub (Phase 4)
# ─────────────────────────────────────────────────────────────────────────────

class SupabaseBackend(StorageBackend):
    """
    Stores call results in a Supabase (Postgres) database.

    To activate: set DATABASE_URL to your Supabase Postgres connection string
    in .env.  Install the extra dependency: pip install ".[supabase]"

    The table schema mirrors SQLiteBackend — you can create it in the Supabase
    SQL editor with the SQL in scripts/init_supabase.sql (generated in Phase 4).
    """

    def __init__(self) -> None:
        try:
            from supabase import create_client  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Supabase backend requires the supabase package. "
                "Install it with: pip install 'voice-agent[supabase]'"
            ) from exc

        settings = get_settings()
        # Supabase requires a project URL + anon key (not a raw Postgres URL).
        # These are separate env vars: SUPABASE_URL and SUPABASE_ANON_KEY.
        import os
        supabase_url = os.environ.get("SUPABASE_URL", "")
        supabase_key = os.environ.get("SUPABASE_ANON_KEY", "")
        if not supabase_url or not supabase_key:
            raise EnvironmentError(
                "Set SUPABASE_URL and SUPABASE_ANON_KEY in .env to use the Supabase backend."
            )
        self._client = create_client(supabase_url, supabase_key)

    def save_call(self, result: dict[str, Any]) -> str:
        row = {
            "call_id":        result["call_id"],
            "started_at":     result.get("started_at"),
            "ended_at":       result.get("ended_at"),
            "collected":      result.get("collected", {}),
            "missing_at_end": result.get("missing_at_end", []),
            "turn_count":     result.get("turn_count", 0),
        }
        self._client.table("agent_calls").upsert(row).execute()
        logger.info("Saved call %s to Supabase", result["call_id"])
        return result["call_id"]

    def get_call(self, call_id: str) -> dict[str, Any] | None:
        response = (
            self._client.table("agent_calls")
            .select("*")
            .eq("call_id", call_id)
            .limit(1)
            .execute()
        )
        if response.data:
            return response.data[0]
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Factory — returns the right backend based on DATABASE_URL
# ─────────────────────────────────────────────────────────────────────────────

def get_storage() -> StorageBackend:
    """
    Return the appropriate storage backend based on DATABASE_URL in .env.
    - Starts with "sqlite://"  → SQLiteBackend
    - Starts with "postgresql://" → SupabaseBackend
    """
    url = get_settings().database_url
    if url.startswith("sqlite"):
        return SQLiteBackend()
    elif url.startswith("postgresql") or url.startswith("postgres"):
        return SupabaseBackend()
    else:
        raise ValueError(
            f"Unknown DATABASE_URL scheme: {url!r}. "
            "Use 'sqlite:///...' or 'postgresql://...'"
        )
