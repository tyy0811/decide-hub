"""Tests for WebSocket connection manager."""

import pytest
from src.serving.ws import ConnectionManager


@pytest.mark.asyncio
async def test_connect_and_disconnect():
    """Manager tracks connected clients."""
    manager = ConnectionManager()
    class FakeWS:
        def __init__(self):
            self.sent = []
            self.accepted = False
        async def accept(self):
            self.accepted = True
        async def send_json(self, data):
            self.sent.append(data)

    ws = FakeWS()
    await manager.connect(ws)
    assert len(manager._connections) == 1

    manager.disconnect(ws)
    assert len(manager._connections) == 0


@pytest.mark.asyncio
async def test_broadcast_sends_to_all():
    """broadcast() sends to all connected clients."""
    manager = ConnectionManager()

    class FakeWS:
        def __init__(self):
            self.sent = []
            self.accepted = False
        async def accept(self):
            self.accepted = True
        async def send_json(self, data):
            self.sent.append(data)

    ws1 = FakeWS()
    ws2 = FakeWS()
    await manager.connect(ws1)
    await manager.connect(ws2)

    await manager.broadcast({"event": "test", "data": "hello"})

    assert len(ws1.sent) == 1
    assert len(ws2.sent) == 1
    assert ws1.sent[0]["event"] == "test"


@pytest.mark.asyncio
async def test_broadcast_removes_failed_connections():
    """Failed sends disconnect the client."""
    manager = ConnectionManager()

    class GoodWS:
        def __init__(self):
            self.sent = []
            self.accepted = False
        async def accept(self):
            self.accepted = True
        async def send_json(self, data):
            self.sent.append(data)

    class BadWS:
        accepted = False
        async def accept(self):
            self.accepted = True
        async def send_json(self, data):
            raise RuntimeError("connection closed")

    good = GoodWS()
    bad = BadWS()
    await manager.connect(good)
    await manager.connect(bad)

    await manager.broadcast({"event": "test"})

    assert len(manager._connections) == 1
    assert len(good.sent) == 1


@pytest.mark.asyncio
async def test_broadcast_noop_when_empty():
    """broadcast() with no connections is a no-op."""
    manager = ConnectionManager()
    await manager.broadcast({"event": "test"})  # Should not raise
