import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

summarizer = None

MAX_TEXT_LENGTH = 10000

class QueryRequest(BaseModel):
    query: str

class SummarizeRequest(BaseModel):
    text: str
    max_length: Optional[int] = 100
    min_length: Optional[int] = 30

@asynccontextmanager
async def lifespan(app: FastAPI):
    global summarizer

    try:
        logger.info("Loading summarization model...")
        summarizer = pipeline(
            task="summarization",
            model="sshleifer/distilbart-cnn-12-6",
            tokenizer="sshleifer/distilbart-cnn-12-6",
            framework="pt",
        )
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load model.")
        raise e

    yield

    logger.info("Shutting down summarization microservice.")


app = FastAPI(
    title="Text Summarization Microservice",
    description="A FastAPI service offering query responses and text summarization using a DistilBART model.",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/query")
async def handle_query(request: Request, q: str):
    logger.info(f"Received query: {q}")
    return {
        "acknowledged_query": q,
        "message": "Query processed successfully!"
    }

@app.post("/summarize")
async def summarize_text(request_payload: SummarizeRequest):
    global summarizer

    try:
        text_to_summarize = request_payload.text.strip()
        if not text_to_summarize:
            raise ValueError("No text provided for summarization")
        
        if len(text_to_summarize) > MAX_TEXT_LENGTH:
            logger.warning(
                f"Text length ({len(text_to_summarize)}) exceeds {MAX_TEXT_LENGTH} characters."
            )
            raise HTTPException(
                status_code=413,
                detail=f"Text length exceeds the maximum allowed size of {MAX_TEXT_LENGTH} characters.",
            )

        logger.info(f"Summarizing text of length {len(text_to_summarize)}")

        max_len = request_payload.max_length if request_payload.max_length else 100
        min_len = request_payload.min_length if request_payload.min_length else 30

        # Generate summary
        summary = summarizer(
            text_to_summarize,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )

        if not summary:
            raise HTTPException(status_code=500, detail="Summarization failed")

        summary_text = summary[0].get("summary_text", "")
        logger.info(f"Summary generated successfully. Summary length: {len(summary_text)}")

        return {
            "original_length": len(text_to_summarize),
            "summary_length": len(summary_text),
            "summary": summary_text
        }
    
    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    
    except HTTPException as he:
        logger.exception(f"HTTPException: {he.detail}")
        raise he
    
    except Exception as e:
        logger.exception("Unknown error during summarization")
        raise HTTPException(status_code=500, detail="Failed to summarize the text.")