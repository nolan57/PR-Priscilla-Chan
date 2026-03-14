#!/usr/bin/env python3
"""
Workflow CLI - Main entry point for batch processing workflows.

Usage:
    python -m workflow.vocoder --input-dir ./raw --output-dir ./data
    python -m workflow.diffsinger --input-dir ./raw --output-dir ./data
    python -m workflow.status --db workflow_state.db

Commands:
    vocoder     - Prepare vocoder training data
    diffsinger  - Prepare DiffSinger training data
    status      - Show workflow status
    reset       - Reset workflow state
"""

import argparse
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Available commands: vocoder, diffsinger, status, reset")
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "vocoder":
        from workflow.vocoder_workflow import main as vocoder_main

        vocoder_main()
    elif command == "diffsinger":
        from workflow.diffsinger_workflow import main as diffsinger_main

        diffsinger_main()
    elif command == "status":
        from workflow.state import StateTracker
        import argparse

        parser = argparse.ArgumentParser(description="Show workflow status")
        parser.add_argument("--db", default="workflow_state.db", help="Database path")
        args = parser.parse_args()

        tracker = StateTracker(args.db)
        print("Workflow Status:")
        print("=" * 40)
        # Query stats
        import sqlite3

        conn = sqlite3.connect(args.db)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT workflow_name, total_files, completed, failed, started_at
            FROM workflow_stats ORDER BY id DESC LIMIT 5
        """)
        for row in cursor.fetchall():
            print(f"Workflow: {row[0]}")
            print(f"  Total: {row[1]}, Completed: {row[2]}, Failed: {row[3]}")
            print(f"  Started: {row[4]}")
            print()
        conn.close()
    elif command == "reset":
        from workflow.state import StateTracker
        import argparse

        parser = argparse.ArgumentParser(description="Reset workflow")
        parser.add_argument("--db", default="workflow_state.db", help="Database path")
        parser.add_argument("--workflow", required=True, help="Workflow name to reset")
        args = parser.parse_args()

        tracker = StateTracker(args.db)
        tracker.reset_workflow(args.workflow)
        print(f"Workflow '{args.workflow}' reset")
    else:
        print(f"Unknown command: {command}")
        print("Available commands: vocoder, diffsinger, status, reset")
        sys.exit(1)


if __name__ == "__main__":
    main()
