import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from main import app, lifespan

client = TestClient(app)

@pytest_asyncio.fixture(scope="module", autouse=True)
async def setup_and_teardown():
    async with lifespan(app):
        yield


@pytest.mark.asyncio
async def test_query_endpoint_success():
    # Test the /query endpoint with a valid query
    response = client.get("/query", params={"q": "Hello World"})
    assert response.status_code == 200
    data = response.json()
    assert "acknowledged_query" in data
    assert data["acknowledged_query"] == "Hello World"


@pytest.mark.asyncio
async def test_summarize_endpoint_success():
    # Test the /summarize endpoint with sample text
    request_body = {
        "text": (
            "The history of infant schools in Great Britain began in 1816, when the first "
            "infant school was founded in New Lanark, Scotland..."
        ),
        "max_length": 50,
        "min_length": 10
    }
    response = client.post("/summarize", json=request_body)
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "original_length" in data
    assert "summary_length" in data


@pytest.mark.asyncio
async def test_summarize_endpoint_no_text():
    # Test the /summarize endpoint with empty text
    request_body = {"text": ""}
    response = client.post("/summarize", json=request_body)
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert data["detail"] == "No text provided for summarization"


@pytest.mark.asyncio
async def test_summarize_endpoint_oversized_payload():
    # Test the request size by sending a large payload
    large_text = "x" * 20000  # Exceeding the 10,000 character limit
    request_body = {"text": large_text}
    response = client.post("/summarize", json=request_body)
    assert response.status_code == 413
    assert b"Text length exceeds the maximum allowed size of" in response.content