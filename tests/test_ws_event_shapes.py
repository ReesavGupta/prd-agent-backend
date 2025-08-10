import pytest

from app.websocket.manager import WebSocketManager


class DummyWs:
    def __init__(self):
        self.sent = []
    async def send_text(self, text: str):
        self.sent.append(text)


@pytest.mark.asyncio
async def test_ws_flowchart_kind_tagging(monkeypatch):
    mgr = WebSocketManager()
    ws = DummyWs()

    # Monkeypatch broadcast to capture messages
    out = []

    async def fake_broadcast(chat_id, message, exclude_websocket=None):
        out.append(message)

    mgr._broadcast_to_chat = fake_broadcast  # type: ignore

    # Simulate connected user mapping
    mgr.connection_users[ws] = {"user_id": "u1", "chat_id": "c1", "websocket_id": "s1"}
    mgr.active_connections["c1"] = set([ws])
    mgr.connection_sessions[ws] = "sess"

    payload = {
        "mode": "flowchart",
        "project_id": "p1",
        "base_prd_markdown": "# PRD\n\n### 1. Product Overview / Purpose\n...",
    }

    await mgr._handle_send_message(ws, "c1", "u1", payload)

    # We expect at least a stream_start and artifacts_preview and ai_response_complete
    types = [m.get("type") for m in out]
    assert "stream_start" in types
    assert "artifacts_preview" in types
    assert "ai_response_complete" in types

    # All should carry kind=flowchart in data
    for m in out:
        data = m.get("data") or {}
        assert data.get("kind") == "flowchart"


