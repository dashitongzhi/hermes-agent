"""Dream Tool — Trigger memory consolidation dreams.

Allows agents to run the DreamEngine for session review and memory consolidation.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict

from tools.registry import registry

log = logging.getLogger(__name__)

DREAM_SCHEMA = {
    "type": "function",
    "function": {
        "name": "dream",
        "description": (
            "Run a memory consolidation dream — reviews recent sessions, "
            "extracts durable insights, and consolidates memory entries. "
            "Useful for periodic memory maintenance."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["run", "history", "config"],
                    "description": "Action to perform. 'run' triggers a dream, "
                                   "'history' shows past dreams, 'config' shows settings.",
                },
                "hours": {
                    "type": "integer",
                    "description": "Lookback hours for session review (default: 24).",
                    "default": 24,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum sessions to review (default: 20).",
                    "default": 20,
                },
            },
            "required": ["action"],
        },
    },
}


def _handle_dream(args: Dict[str, Any]) -> str:
    """Handle dream tool calls."""
    action = args.get("action", "run")
    hours = args.get("hours", 24)
    limit = args.get("limit", 20)

    try:
        from hermes_constants import get_hermes_home
        from agent.dream_engine import DreamEngine
        from hermes_state import SessionDB

        hermes_home = get_hermes_home()

        if action == "config":
            return json.dumps({
                "status": "ok",
                "config": {
                    "lookback_hours": hours,
                    "max_sessions": limit,
                    "memory_dir": str(hermes_home),
                }
            }, indent=2)

        if action == "history":
            dream_log = hermes_home / "dreams.json"
            if dream_log.exists():
                with open(dream_log, "r") as f:
                    history = json.load(f)
                return json.dumps({"status": "ok", "dreams": history[-10:]}, indent=2)
            return json.dumps({"status": "ok", "dreams": [], "message": "No dreams yet."})

        if action == "run":
            # Initialize components
            db_path = hermes_home / "state.db"
            session_db = SessionDB(str(db_path))
            engine = DreamEngine(session_db, hermes_home)

            # Gather sessions
            sessions = engine.gather_sessions(hours=hours, limit=limit)
            if not sessions:
                return json.dumps({
                    "status": "ok",
                    "message": f"No sessions found in the last {hours} hours."
                })

            # Build and return the dream prompt (the agent will execute it)
            prompt = engine.extract_insights(sessions)
            return json.dumps({
                "status": "ok",
                "session_count": len(sessions),
                "prompt_preview": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                "message": f"Found {len(sessions)} sessions to review. "
                           "Dream consolidation would run here.",
            }, indent=2)

        return json.dumps({"status": "error", "message": f"Unknown action: {action}"})

    except Exception as e:
        log.error("Dream tool error: %s", e)
        return json.dumps({"status": "error", "message": str(e)})


# Register the tool
registry.register("dream", "memory", DREAM_SCHEMA, _handle_dream)
