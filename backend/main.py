from fastapi import (
    FastAPI,
    HTTPException,
    Request
)

from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

import os

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

    print(
        "Loading Bharat AI resources..."
    )

    # Load ChromaDB + embedding model
    retrieve.get_vector_store()

    print(
        "Bharat AI resources loaded."
    )


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):

    question: str

    # True when user used microphone
    voice: bool = False


# ============================================================
# RESPONSE MODEL
# ============================================================

class ChatResponse(BaseModel):

    answer: str

    # Audio URL when voice=True
    audio_url: str | None = None


# ============================================================
# ASK AI
# ============================================================

@app.post("/ask", response_model=ChatResponse)
async def ask_question(
    request: ChatRequest,
    http_request: Request
):
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        # RAG + AI
        answer = retrieve.answer_question(
            request.question
        )

        # Browser will handle TTS.
        return ChatResponse(
            answer=answer,
            audio_url=None
        )

    except Exception as e:
        print(f"Bharat AI error: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if (
        not request.question
        or not request.question.strip()
    ):

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        # ----------------------------------------------------
        # RAG + GEMINI
        # ----------------------------------------------------

        answer = retrieve.answer_question(
            request.question
        )

        # ----------------------------------------------------
        # TEXT ONLY
        # ----------------------------------------------------

        if not request.voice:

            return ChatResponse(
                answer=answer,
                audio_url=None
            )

        # ----------------------------------------------------
        # VOICE OUTPUT
        # ----------------------------------------------------

        audio_filename = generate_voice(
            answer
        )

        # ----------------------------------------------------
        # Build public audio URL
        # ----------------------------------------------------

        audio_url = (
            str(http_request.base_url).rstrip("/")
            + "/audio/"
            + audio_filename
        )

        return ChatResponse(
            answer=answer,
            audio_url=audio_url
        )

    except Exception as e:

        print(
            f"Bharat AI error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "ok",
        "service": "Bharat AI"
    }