#!/usr/bin/env python3
"""
State Tracker - SQLite-based state tracking for workflow processing.
Handles breakpoint resume and progress tracking.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional


STAGES = [
    "raw",  # Original input file
    "separated",  # After vocal separation
    "cleaned",  # After audio cleaning
    "f0_extracted",  # After F0 extraction
    "aligned",  # After MFA alignment
    "npz_ready",  # Ready for NPZ building
    "completed",  # Fully processed
]


class StateTracker:
    def __init__(self, db_path: str = "workflow_state.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT,
                updated_at TIMESTAMP NOT NULL,
                UNIQUE(filepath)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workflow_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_name TEXT NOT NULL,
                total_files INTEGER NOT NULL,
                completed INTEGER NOT NULL,
                failed INTEGER NOT NULL,
                started_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        """)

        conn.commit()
        conn.close()

    def init_workflow(self, workflow_name: str, total_files: int):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO workflow_stats (workflow_name, total_files, completed, failed, started_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (workflow_name, total_files, 0, 0, datetime.now(), datetime.now()),
        )

        conn.commit()
        conn.close()

    def update_file(
        self,
        filepath: str,
        stage: str,
        status: str = "completed",
        error_message: Optional[str] = None,
    ):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO files (filepath, filename, stage, status, error_message, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(filepath) DO UPDATE SET
                stage = excluded.stage,
                status = excluded.status,
                error_message = excluded.error_message,
                updated_at = excluded.updated_at
        """,
            (
                filepath,
                Path(filepath).name,
                stage,
                status,
                error_message,
                datetime.now(),
            ),
        )

        conn.commit()
        conn.close()

    def get_file_stage(self, filepath: str) -> Optional[str]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT stage, status FROM files WHERE filepath = ?
        """,
            (filepath,),
        )

        result = cursor.fetchone()
        conn.close()

        if result:
            return result[0] if result[1] == "completed" else None
        return None

    def is_completed(self, filepath: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT status FROM files WHERE filepath = ?
        """,
            (filepath,),
        )

        result = cursor.fetchone()
        conn.close()

        return result is not None and result[0] == "completed"

    def get_pending_files(self, stage: str) -> list:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT filepath FROM files 
            WHERE stage != ? OR status != 'completed'
        """,
            (stage,),
        )

        results = [row[0] for row in cursor.fetchall()]
        conn.close()

        return results

    def get_stats(self, workflow_name: str) -> dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT total_files, completed, failed FROM workflow_stats
            WHERE workflow_name = ? ORDER BY id DESC LIMIT 1
        """,
            (workflow_name,),
        )

        result = cursor.fetchone()
        conn.close()

        if result:
            return {"total": result[0], "completed": result[1], "failed": result[2]}
        return {"total": 0, "completed": 0, "failed": 0}

    def increment_completed(self, workflow_name: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE workflow_stats SET 
                completed = completed + 1,
                updated_at = ?
            WHERE workflow_name = ? AND id = (SELECT MAX(id) FROM workflow_stats WHERE workflow_name = ?)
        """,
            (datetime.now(), workflow_name, workflow_name),
        )

        conn.commit()
        conn.close()

    def increment_failed(self, workflow_name: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE workflow_stats SET 
                failed = failed + 1,
                updated_at = ?
            WHERE workflow_name = ? AND id = (SELECT MAX(id) FROM workflow_stats WHERE workflow_name = ?)
        """,
            (datetime.now(), workflow_name, workflow_name),
        )

        conn.commit()
        conn.close()

    def reset_workflow(self, workflow_name: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM files WHERE filepath IN (
                SELECT filepath FROM files 
                WHERE stage != 'completed' OR status != 'completed'
            )
        """)

        cursor.execute(
            """
            DELETE FROM workflow_stats WHERE workflow_name = ?
        """,
            (workflow_name,),
        )

        conn.commit()
        conn.close()


def main():
    tracker = StateTracker()
    print("State tracker initialized")


if __name__ == "__main__":
    main()
