from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import retrieve

app = FastAPI(title="Personal AI Assistant API")


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_resources():
    # Load the embedding model + Chroma DB once, when the server boots,
    # so the first real user request isn't the one paying the load cost.
    retrieve.get_vector_store()


# Request model
class ChatRequest(BaseModel):
    question: str


# Response model
class ChatResponse(BaseModel):
    answer: str


# AI question endpoint
@app.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):

    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    answer = retrieve.answer_question(request.question)

    return ChatResponse(answer=answer)


# Health check
@app.get("/health")
def health_check():
    return {"status": "ok"}