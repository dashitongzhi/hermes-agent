"""Tests for Dream Engine."""
import json
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from agent.dream_engine import DreamEngine, DreamResult


@pytest.fixture
def mock_session_db():
    db = MagicMock()
    db.search_sessions.return_value = [
        {"id": "s1", "title": "Test Session 1", "created_at": time.time() - 3600},
        {"id": "s2", "title": "Test Session 2", "created_at": time.time() - 7200},
    ]
    return db


@pytest.fixture
def engine(mock_session_db, tmp_path):
    return DreamEngine(mock_session_db, tmp_path)


def test_gather_sessions(engine, mock_session_db):
    sessions = engine.gather_sessions(hours=24, limit=10)
    assert len(sessions) == 2
    mock_session_db.search_sessions.assert_called_once()


def test_gather_sessions_empty(engine, mock_session_db):
    mock_session_db.search_sessions.return_value = []
    sessions = engine.gather_sessions()
    assert sessions == []


def test_extract_insights(engine):
    sessions = [
        {"id": "s1", "title": "Memory Setup", "created_at": time.time()},
        {"id": "s2", "title": "Debug Session", "created_at": time.time()},
    ]
    prompt = engine.extract_insights(sessions)
    assert "dream mode" in prompt.lower()
    assert "Memory Setup" in prompt
    assert "Debug Session" in prompt


def test_extract_insights_empty(engine):
    prompt = engine.extract_insights([])
    assert "No recent sessions" in prompt


def test_dream_result_serialization():
    result = DreamResult(
        session_count=5,
        insights="Test insights",
        memory_updates=["/path/to/MEMORY.md"],
        duration_seconds=10.5,
    )
    engine = DreamEngine(MagicMock(), Path("/tmp"))
    data = engine.to_dict(result)
    assert data["session_count"] == 5
    assert data["insights"] == "Test insights"
    assert data["duration_seconds"] == 10.5


def test_apply_consolidation(engine, tmp_path):
    memory_file = tmp_path / "MEMORY.md"
    memory_file.write_text("# Memory\n")
    user_file = tmp_path / "USER.md"
    user_file.write_text("# User\n")

    consolidation = "## Environment Facts\n- User prefers Python 3.12\n- macOS ARM64"
    updated = engine.apply_consolidation(consolidation)
    assert len(updated) >= 1
