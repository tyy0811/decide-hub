"""Entity collector — async httpx client for fetching entities from data source."""

import httpx


async def fetch_entities(url: str, timeout: float = 30.0) -> list[dict]:
    """Fetch entities from a data source URL.

    In tests: targets mock_lead_api.py.
    In production: would point at a real API or webhook.
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("leads", data.get("entities", []))
