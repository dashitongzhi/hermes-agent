"""Outcomes Engine — Rubric-based grading for agent outputs.

Provides LLM-as-judge grading against configurable rubrics,
with a review loop that retries when output quality is below threshold.

Config:
    Rubrics stored in ~/.hermes/rubrics/<name>.yaml
    Outcomes stored in state.db outcomes table
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class RubricCriterion:
    """A single grading criterion."""
    criterion: str
    weight: float
    description: str


@dataclass
class Rubric:
    """A grading rubric with weighted criteria."""
    name: str
    description: str
    criteria: List[RubricCriterion]
    pass_threshold: float = 0.7


@dataclass
class Outcome:
    """Result of grading an output against a rubric."""
    rubric_name: str
    scores: Dict[str, float]  # criterion -> score (0.0-1.0)
    total_score: float
    passed: bool
    feedback: str
    session_id: str = ""
    timestamp: float = field(default_factory=time.time)


class OutcomeEngine:
    """Grades agent outputs against rubrics."""

    def __init__(self, rubrics_dir: Path, session_db=None):
        self._rubrics_dir = rubrics_dir
        self._session_db = session_db

    def load_rubric(self, name: str) -> Optional[Rubric]:
        """Load rubric from ~/.hermes/rubrics/<name>.yaml."""
        rubric_file = self._rubrics_dir / f"{name}.yaml"
        if not rubric_file.exists():
            # Try .yml
            rubric_file = self._rubrics_dir / f"{name}.yml"
        if not rubric_file.exists():
            log.warning("Rubric not found: %s", name)
            return None

        try:
            import yaml
            with open(rubric_file, "r") as f:
                data = yaml.safe_load(f)
        except ImportError:
            # Fallback: parse simple YAML manually
            data = self._parse_simple_yaml(rubric_file)

        criteria = []
        for c in data.get("criteria", []):
            criteria.append(RubricCriterion(
                criterion=c.get("criterion", c.get("name", "")),
                weight=c.get("weight", 1.0),
                description=c.get("description", ""),
            ))

        return Rubric(
            name=data.get("name", name),
            description=data.get("description", ""),
            criteria=criteria,
            pass_threshold=data.get("pass_threshold", 0.7),
        )

    def list_rubrics(self) -> List[str]:
        """List available rubric names."""
        if not self._rubrics_dir.exists():
            return []
        return [
            f.stem for f in sorted(self._rubrics_dir.glob("*.yaml"))
        ] + [
            f.stem for f in sorted(self._rubrics_dir.glob("*.yml"))
        ]

    def grade(self, rubric: Rubric, output: str, context: str = "") -> Outcome:
        """Grade output against rubric criteria using LLM-as-judge.

        This builds a grading prompt. In actual use, an AIAgent would execute it.
        For tool usage, we return a structured grading request.
        """
        criteria_text = "\n".join(
            f"- {c.criterion} (weight: {c.weight}): {c.description}"
            for c in rubric.criteria
        )

        grading_prompt = (
            f"Grade the following output against the rubric \"{rubric.name}\".\n\n"
            f"**Rubric Criteria:**\n{criteria_text}\n\n"
            f"**Pass Threshold:** {rubric.pass_threshold}\n\n"
            f"**Context:** {context}\n\n"
            f"**Output to Grade:**\n{output}\n\n"
            f"Respond with a JSON object:\n"
            f'{{"scores": {{"criterion_name": 0.0-1.0, ...}}, '
            f'"total_score": 0.0-1.0, "feedback": "detailed feedback"}}'
        )

        # For now, return the prompt for an agent to execute
        # In a full implementation, this would spawn a grading agent
        return Outcome(
            rubric_name=rubric.name,
            scores={c.criterion: 0.0 for c in rubric.criteria},
            total_score=0.0,
            passed=False,
            feedback="Grading requires LLM execution. Use the outcomes tool.",
        )

    def review_loop(self, rubric: Rubric, task_fn, max_retries: int = 2,
                    context: str = "") -> Outcome:
        """Run task, grade output, retry with feedback if below threshold.

        Args:
            rubric: The rubric to grade against.
            task_fn: Callable that returns the output string.
            max_retries: Maximum retry attempts.
            context: Additional context for grading.
        """
        for attempt in range(max_retries + 1):
            output = task_fn()
            outcome = self.grade(rubric, output, context)

            if outcome.passed:
                log.info("Passed on attempt %d/%d", attempt + 1, max_retries + 1)
                return outcome

            if attempt < max_retries:
                log.info(
                    "Failed attempt %d/%d (score=%.2f, threshold=%.2f). Retrying...",
                    attempt + 1, max_retries + 1,
                    outcome.total_score, rubric.pass_threshold,
                )

        return outcome

    def store_outcome(self, outcome: Outcome) -> bool:
        """Persist outcome to state.db."""
        if not self._session_db:
            return False
        try:
            if hasattr(self._session_db, 'store_outcome'):
                self._session_db.store_outcome(
                    session_id=outcome.session_id,
                    rubric_name=outcome.rubric_name,
                    total_score=outcome.total_score,
                    passed=outcome.passed,
                    scores_json=json.dumps(outcome.scores),
                    feedback=outcome.feedback,
                    timestamp=outcome.timestamp,
                )
                return True
        except Exception as e:
            log.error("Failed to store outcome: %s", e)
        return False

    def get_outcomes(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent outcomes."""
        if not self._session_db:
            return []
        try:
            if hasattr(self._session_db, 'get_outcomes'):
                return self._session_db.get_outcomes(limit=limit)
        except Exception as e:
            log.error("Failed to get outcomes: %s", e)
        return []

    def _parse_simple_yaml(self, path: Path) -> Dict:
        """Minimal YAML parser for rubric files (fallback when PyYAML unavailable)."""
        import re
        data = {"criteria": []}
        current_criterion = None

        with open(path, "r") as f:
            for line in f:
                line = line.rstrip()
                if not line or line.startswith("#"):
                    continue

                # Top-level key: value
                match = re.match(r"^(\w+):\s*(.+)$", line)
                if match and not line.startswith("  "):
                    key, val = match.groups()
                    if key == "pass_threshold":
                        data[key] = float(val)
                    elif key != "criteria":
                        data[key] = val.strip().strip('"').strip("'")

                # Criterion block
                if line.strip().startswith("- criterion:"):
                    if current_criterion:
                        data["criteria"].append(current_criterion)
                    name = line.split(":", 1)[1].strip().strip('"').strip("'")
                    current_criterion = {"criterion": name, "weight": 1.0, "description": ""}
                elif current_criterion:
                    m = re.match(r"^\s+(\w+):\s*(.+)$", line)
                    if m:
                        k, v = m.groups()
                        if k == "weight":
                            current_criterion[k] = float(v)
                        else:
                            current_criterion[k] = v.strip().strip('"').strip("'")

            if current_criterion:
                data["criteria"].append(current_criterion)

        return data

    def to_dict(self, outcome: Outcome) -> Dict[str, Any]:
        """Serialize Outcome to dict."""
        return {
            "rubric_name": outcome.rubric_name,
            "scores": outcome.scores,
            "total_score": outcome.total_score,
            "passed": outcome.passed,
            "feedback": outcome.feedback,
            "session_id": outcome.session_id,
            "timestamp": outcome.timestamp,
        }
