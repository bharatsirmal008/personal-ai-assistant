from fastapi import (
    FastAPI,
    HTTPException,
    Request
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from fastapi.responses import (
    StreamingResponse
)

from pydantic import BaseModel

import os
import json

import retrieve

from tts import generate_voice


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Bharat AI Personal Assistant API"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

AUDIO_DIR = os.path.join(
    BASE_DIR,
    "audio"
)


# ============================================================
# CREATE AUDIO DIRECTORY
# ============================================================

os.makedirs(
    AUDIO_DIR,
    exist_ok=True
)


# ============================================================
# STATIC AUDIO
# ============================================================

app.mount(
    "/audio",
    StaticFiles(
        directory=AUDIO_DIR
    ),
    name="audio"
)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def load_resources():

    print()
    print("=" * 60)
    print("BHARAT AI STARTING")
    print("=" * 60)

    retrieve.get_vector_store()

    print(
        "Bharat AI resources loaded."
    )

    print("=" * 60)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):

    question: str

    voice: bool = False

    session_id: str = "default"


# ============================================================
# STREAM GENERATOR
# ============================================================

def generate_stream(
    question: str,
    session_id: str
):

    try:

        for chunk in retrieve.stream_answer(
            question,
            session_id
        ):

            if not chunk:
                continue

            # Send every chunk immediately
            yield (
                json.dumps({
                    "token": chunk
                })
                + "\n"
            )

    except Exception as e:

        print(
            f"Streaming error: {e}"
        )

        yield (
            json.dumps({
                "error":
                    "I'm having trouble processing "
                    "that request right now."
            })
            + "\n"
        )


# ============================================================
# ASK AI — STREAMING
# ============================================================

@app.post("/ask")
async def ask_question(
    request: ChatRequest,
    http_request: Request
):

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    question = (
        request.question or ""
    ).strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )


    print()
    print("=" * 60)
    print("NEW STREAMING QUESTION")
    print("=" * 60)

    print(
        f"Question: {question}"
    )

    print(
        f"Session: {request.session_id}"
    )

    print(
        f"Voice: {request.voice}"
    )

    # --------------------------------------------------------
    # STREAM RESPONSE
    # --------------------------------------------------------

    return StreamingResponse(

        generate_stream(
            question,
            request.session_id
        ),

        media_type="application/x-ndjson",

        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "ok",
        "service": "Bharat AI",
        "streaming": True
    }