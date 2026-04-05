"""WebSocket connection manager — in-process broadcast for live updates.

Single-process design. Multi-process would need Redis pub/sub — that's a
deployment concern, not a design concern. See DECISIONS.md.
"""

from fastapi import WebSocket


class ConnectionManager:
    """Manages WebSocket connections and broadcasts events."""

    def __init__(self):
        self._connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)

    async def broadcast(self, message: dict) -> None:
        """Send message to all connected clients. Fire-and-forget.

        Failed sends silently remove the connection.
        """
        dead = []
        for ws in self._connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._connections.discard(ws)


# Global instance used by orchestrator and endpoint
ws_manager = ConnectionManager()
