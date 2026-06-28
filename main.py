"""
FastAPI entrypoint for the Ford vehicle intelligence RAG agent.
"""

import logging
import traceback
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

load_dotenv()

from agent.graph import get_agent, get_fallback_agent

app = FastAPI(
    title="Ford Vehicle Insight Agent",
    description="RAG + tool-calling agent over NHTSA public vehicle safety data",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8001",
        "http://localhost:8001",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    thread_id: str
    fallback: bool = False
    notice: Optional[str] = None


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    agent = get_agent()

    config = {"configurable": {"thread_id": thread_id}}
    fallback_used = False
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": req.message}]},
            config=config,
        )
    except Exception as e:
        error_text = str(e)
        if any(token in error_text for token in ["429", "rate_limit_exceeded", "Rate limit", "rate limit"]):
            logging.warning(
                "Rate limit hit for thread %s; falling back to basic assistant response. Error: %s",
                thread_id,
                error_text,
            )
            fallback = get_fallback_agent()
            result = fallback.invoke({"messages": [{"role": "user", "content": req.message}]})
            fallback_used = True
        else:
            logging.error("Agent error for thread %s:\n%s", thread_id, traceback.format_exc())
            raise HTTPException(status_code=500, detail=error_text)

    # Extract the last AI message from the graph output
    messages = result.get("messages", [])
    ai_messages = [m for m in messages if hasattr(m, "type") and m.type == "ai"]
    if not ai_messages:
        raise HTTPException(status_code=500, detail="Agent returned no response")

    content = ai_messages[-1].content
    # Gemini returns content as a list of parts; extract text
    if isinstance(content, list):
        content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)

    notice = None
    if fallback_used:
        if any(marker in content for marker in [
            "Semantic complaint search is currently unavailable",
            "Live complaint lookup is unavailable",
            "I am providing relevant live Ford vehicle safety information instead",
        ]):
            notice = "Knowledge base unavailable; returning live NHTSA data or fallback assistance."
        else:
            notice = "Model rate-limited; using fallback response."
    elif "Information is not available in the local knowledge base at the moment." in content:
        notice = "Knowledge-base entry not found; returning relevant live NHTSA data."

    return ChatResponse(
        response=content,
        thread_id=thread_id,
        fallback=fallback_used,
        notice=notice,
    )
