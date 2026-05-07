"""Tests for the outbound webhook dispatcher (agent/webhook_dispatcher.py)."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def tmp_webhooks_file(tmp_path, monkeypatch):
    """Redirect the webhooks JSON storage to a temp directory."""
    wh_path = tmp_path / "webhooks.json"
    monkeypatch.setattr(
        "agent.webhook_dispatcher._webhooks_path", lambda: wh_path
    )
    return wh_path


@pytest.fixture()
def dispatcher(tmp_webhooks_file):
    """Return a fresh WebhookDispatcher using the temp storage."""
    from agent.webhook_dispatcher import WebhookDispatcher
    return WebhookDispatcher(storage_path=tmp_webhooks_file)


class TestWebhookRegistration:
    """Test webhook register/unregister/list operations."""

    def test_register_returns_id(self, dispatcher):
        wh_id = dispatcher.register(
            url="https://example.com/hook",
            events=["task:completed"],
        )
        assert wh_id is not None
        assert isinstance(wh_id, str)
        assert len(wh_id) > 0

    def test_register_and_list(self, dispatcher):
        dispatcher.register(
            url="https://example.com/hook1",
            events=["task:completed"],
        )
        dispatcher.register(
            url="https://example.com/hook2",
            events=["cron:completed", "cron:failed"],
        )
        webhooks = dispatcher.list_webhooks()
        assert len(webhooks) == 2
        urls = {wh["url"] for wh in webhooks}
        assert "https://example.com/hook1" in urls
        assert "https://example.com/hook2" in urls

    def test_unregister(self, dispatcher):
        wh_id = dispatcher.register(
            url="https://example.com/hook",
            events=["task:completed"],
        )
        assert len(dispatcher.list_webhooks()) == 1
        result = dispatcher.unregister(wh_id)
        assert result is True
        assert len(dispatcher.list_webhooks()) == 0

    def test_unregister_nonexistent(self, dispatcher):
        result = dispatcher.unregister("nonexistent-id")
        assert result is False

    def test_secret_redacted_in_list(self, dispatcher):
        dispatcher.register(
            url="https://example.com/hook",
            events=["task:completed"],
            secret="my-secret-key",
        )
        webhooks = dispatcher.list_webhooks()
        assert len(webhooks) == 1
        assert webhooks[0]["secret"] == "***"

    def test_get_by_id(self, dispatcher):
        wh_id = dispatcher.register(
            url="https://example.com/hook",
            events=["task:completed"],
            secret="test-secret",
        )
        cfg = dispatcher.get(wh_id)
        assert cfg is not None
        assert cfg.url == "https://example.com/hook"
        assert cfg.secret == "test-secret"
        assert cfg.events == ["task:completed"]

    def test_persistence(self, tmp_webhooks_file):
        """Webhooks survive a new dispatcher instance."""
        from agent.webhook_dispatcher import WebhookDispatcher

        d1 = WebhookDispatcher(storage_path=tmp_webhooks_file)
        wh_id = d1.register(
            url="https://example.com/persist",
            events=["task:*"],
        )
        # New instance should load from file
        d2 = WebhookDispatcher(storage_path=tmp_webhooks_file)
        assert len(d2.list_webhooks()) == 1
        cfg = d2.get(wh_id)
        assert cfg is not None
        assert cfg.url == "https://example.com/persist"


class TestEventMatching:
    """Test the event matching logic."""

    def test_exact_match(self, dispatcher):
        assert dispatcher._event_matches("task:completed", ["task:completed"]) is True

    def test_no_match(self, dispatcher):
        assert dispatcher._event_matches("task:completed", ["cron:completed"]) is False

    def test_wildcard_match(self, dispatcher):
        assert dispatcher._event_matches("task:completed", ["task:*"]) is True
        assert dispatcher._event_matches("task:failed", ["task:*"]) is True

    def test_star_matches_all(self, dispatcher):
        assert dispatcher._event_matches("task:completed", ["*"]) is True
        assert dispatcher._event_matches("cron:failed", ["*"]) is True

    def test_multiple_patterns(self, dispatcher):
        patterns = ["task:completed", "cron:*"]
        assert dispatcher._event_matches("task:completed", patterns) is True
        assert dispatcher._event_matches("cron:completed", patterns) is True
        assert dispatcher._event_matches("cron:failed", patterns) is True
        assert dispatcher._event_matches("task:failed", patterns) is False


class TestHMACSigning:
    """Test HMAC-SHA256 payload signing."""

    def test_sign_format(self):
        from agent.webhook_dispatcher import WebhookDispatcher
        sig = WebhookDispatcher._sign_payload(b"hello", "secret")
        assert sig.startswith("sha256=")
        assert len(sig) == len("sha256=") + 64  # hex digest

    def test_sign_deterministic(self):
        from agent.webhook_dispatcher import WebhookDispatcher
        sig1 = WebhookDispatcher._sign_payload(b"hello", "secret")
        sig2 = WebhookDispatcher._sign_payload(b"hello", "secret")
        assert sig1 == sig2

    def test_sign_different_secret(self):
        from agent.webhook_dispatcher import WebhookDispatcher
        sig1 = WebhookDispatcher._sign_payload(b"hello", "secret1")
        sig2 = WebhookDispatcher._sign_payload(b"hello", "secret2")
        assert sig1 != sig2

    def test_sign_different_payload(self):
        from agent.webhook_dispatcher import WebhookDispatcher
        sig1 = WebhookDispatcher._sign_payload(b"hello", "secret")
        sig2 = WebhookDispatcher._sign_payload(b"world", "secret")
        assert sig1 != sig2


class TestRetryPolicy:
    """Test retry policy backoff calculations."""

    def test_default_delay(self):
        from agent.webhook_dispatcher import RetryPolicy
        rp = RetryPolicy()
        assert rp.delay_for_attempt(0) == 1.0  # 2^0 = 1
        assert rp.delay_for_attempt(1) == 2.0  # 2^1 = 2
        assert rp.delay_for_attempt(2) == 4.0  # 2^2 = 4

    def test_backoff_max_cap(self):
        from agent.webhook_dispatcher import RetryPolicy
        rp = RetryPolicy(backoff_base=10, backoff_max=5.0)
        assert rp.delay_for_attempt(0) == 1.0   # 10^0 = 1 (min(1, 5) = 1)
        assert rp.delay_for_attempt(1) == 5.0   # min(10, 5) = 5
        assert rp.delay_for_attempt(2) == 5.0   # min(100, 5) = 5

    def test_custom_base(self):
        from agent.webhook_dispatcher import RetryPolicy
        rp = RetryPolicy(backoff_base=3, backoff_max=100)
        assert rp.delay_for_attempt(0) == 1.0  # 3^0
        assert rp.delay_for_attempt(1) == 3.0  # 3^1
        assert rp.delay_for_attempt(2) == 9.0  # 3^2


class TestDispatch:
    """Test dispatching events to webhooks."""

    def _mock_requests(self):
        """Create a mock requests module with configurable post."""
        mock_module = MagicMock()
        return mock_module

    def test_dispatch_calls_matching_webhooks(self, dispatcher):
        dispatcher.register(
            url="https://example.com/hook",
            events=["task:completed"],
        )
        mock_req = self._mock_requests()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_req.post.return_value = mock_resp

        with patch("builtins.__import__", side_effect=lambda name, *a, **kw: mock_req if name == "requests" else __import__(name, *a, **kw)):
            results = dispatcher.dispatch("task:completed", {"task_id": "abc"})

        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["status_code"] == 200
        mock_req.post.assert_called_once()
        # Verify the payload was JSON with correct event
        call_args = mock_req.post.call_args
        body = json.loads(call_args.kwargs.get("data") or call_args[1].get("data", b""))
        assert body["event"] == "task:completed"
        assert body["payload"]["task_id"] == "abc"

    def test_dispatch_skips_non_matching(self, dispatcher):
        dispatcher.register(
            url="https://example.com/hook",
            events=["task:completed"],
        )
        results = dispatcher.dispatch("cron:completed", {"job_id": "xyz"})
        assert len(results) == 0

    def test_dispatch_retries_on_failure(self, dispatcher):
        dispatcher.register(
            url="https://example.com/hook",
            events=["task:*"],
            retry_policy={"max_retries": 3, "backoff_base": 1, "backoff_max": 1},
        )
        mock_req = self._mock_requests()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_req.post.return_value = mock_resp

        with patch("builtins.__import__", side_effect=lambda name, *a, **kw: mock_req if name == "requests" else __import__(name, *a, **kw)), \
             patch("agent.webhook_dispatcher.time.sleep") as mock_sleep:
            results = dispatcher.dispatch("task:completed", {"task_id": "abc"})

        assert len(results) == 1
        assert results[0]["success"] is False
        # Should have tried 3 times
        assert mock_req.post.call_count == 3
        # Should have slept 2 times (between attempts)
        assert mock_sleep.call_count == 2

    def test_dispatch_succeeds_on_second_attempt(self, dispatcher):
        dispatcher.register(
            url="https://example.com/hook",
            events=["task:*"],
            retry_policy={"max_retries": 3, "backoff_base": 1, "backoff_max": 1},
        )
        mock_req = self._mock_requests()
        fail_resp = MagicMock()
        fail_resp.status_code = 500
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        mock_req.post.side_effect = [fail_resp, ok_resp]

        with patch("builtins.__import__", side_effect=lambda name, *a, **kw: mock_req if name == "requests" else __import__(name, *a, **kw)), \
             patch("agent.webhook_dispatcher.time.sleep"):
            results = dispatcher.dispatch("task:completed", {"task_id": "abc"})

        assert len(results) == 1
        assert results[0]["success"] is True
        assert mock_req.post.call_count == 2

    def test_dispatch_with_hmac_signature(self, dispatcher):
        dispatcher.register(
            url="https://example.com/hook",
            events=["task:completed"],
            secret="my-test-secret",
        )
        mock_req = self._mock_requests()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_req.post.return_value = mock_resp

        with patch("builtins.__import__", side_effect=lambda name, *a, **kw: mock_req if name == "requests" else __import__(name, *a, **kw)):
            dispatcher.dispatch("task:completed", {"task_id": "abc"})

        call_args = mock_req.post.call_args
        headers = call_args.kwargs.get("headers") or call_args[1].get("headers", {})
        assert "X-Hermes-Signature" in headers
        assert headers["X-Hermes-Signature"].startswith("sha256=")

    def test_dispatch_no_webhooks(self, dispatcher):
        """Dispatching when no webhooks are registered returns empty results."""
        results = dispatcher.dispatch("task:completed", {"task_id": "abc"})
        assert results == []
