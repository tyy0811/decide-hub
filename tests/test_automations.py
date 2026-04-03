import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from tests.mock_lead_api import MOCK_LEADS
from src.automations.crawler import fetch_entities


@pytest.mark.asyncio
async def test_fetch_entities():
    """Crawler fetches entities from a URL."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"leads": MOCK_LEADS}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        entities = await fetch_entities("http://localhost:9999/leads")

    assert len(entities) == 5
    assert entities[0]["entity_id"] == "lead_001"
