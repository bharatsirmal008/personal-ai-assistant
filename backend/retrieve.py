import os
import json
from typing import Generator

from dotenv import load_dotenv
from google import genai
from groq import Groq

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from prompt import prompt_template

import memory


# ============================================================
# ENVIRONMENT
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ENV_FILE = os.path.join(
    BASE_DIR,
    ".env"
)

load_dotenv(ENV_FILE)


# ============================================================
# PATHS
# ============================================================

DB_DIR = os.path.join(
    BASE_DIR,
    "chroma_db"
)


# ============================================================
# RAG SETTINGS
# ============================================================

TOP_K = 6

THRESHOLD = 1.7


# ============================================================
# GEMINI
# ============================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash"
)


# ============================================================
# GROQ
# ============================================================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)


# ============================================================
# GLOBAL CACHE
# ============================================================

_embeddings = None
_vector_store = None

_gemini_client = None
_groq_client = None


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_gemini_client():

    global _gemini_client

    if _gemini_client is None:

        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured."
            )

        _gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

    return _gemini_client


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client():

    global _groq_client

    if _groq_client is None:

        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not configured."
            )

        _groq_client = Groq(
            api_key=GROQ_API_KEY
        )

    return _groq_client


# ============================================================
# VECTOR STORE
# ============================================================

def get_vector_store():

    global _embeddings
    global _vector_store

    if _vector_store is None:

        print(
            "Loading embedding model and ChromaDB..."
        )

        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        _vector_store = Chroma(
            persist_directory=DB_DIR,
            embedding_function=_embeddings
        )

        print(
            "Embedding model and ChromaDB loaded."
        )

    return _vector_store


# ============================================================
# QUOTA ERROR
# ============================================================

def is_quota_error(error: Exception) -> bool:

    error_text = str(error).lower()

    quota_keywords = [
        "429",
        "resource_exhausted",
        "quota exceeded",
        "quota",
        "rate limit",
        "rate_limit"
    ]

    return any(
        keyword in error_text
        for keyword in quota_keywords
    )


# ============================================================
# DIRECT JOJO RESPONSES
# ============================================================

def get_jojo_direct_response(
    user_question: str
):

    q = user_question.strip().lower()

    # --------------------------------------------------------
    # Greetings
    # --------------------------------------------------------

    if q in [
        "hi",
        "hello",
        "hey",
        "hi jojo",
        "hello jojo",
        "hey jojo"
    ]:

        return (
            "Hi! I'm Jojo. How can I help you?"
        )

    # --------------------------------------------------------
    # Name
    # --------------------------------------------------------

    if q in [
        "what is your name",
        "what's your name",
        "what is ur name",
        "your name",
        "tell me your name"
    ]:

        return (
            "My name is Jojo."
        )

    # --------------------------------------------------------
    # Who are you
    # --------------------------------------------------------

    if q in [
        "who are you",
        "who r you",
        "who is jojo",
        "who's jojo",
        "tell me about yourself"
    ]:

        return (
            "I'm Jojo, Bharat Sirmal's personal AI assistant."
        )

    # --------------------------------------------------------
    # Boss
    # --------------------------------------------------------

    if q in [
        "who is your boss",
        "who's your boss",
        "who is ur boss",
        "who is your owner",
        "who owns you"
    ]:

        return (
            "Bharat Sirmal is my boss."
        )

    return None


# ============================================================
# BUILD RAG CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    for index, (doc, score) in enumerate(
        results,
        start=1
    ):

        source = doc.metadata.get(
            "source",
            "Unknown source"
        )

        context_parts.append(
            f"""
--- SOURCE {index}: {source} ---
Relevance score: {score:.4f}

{doc.page_content}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# MEMORY
# ============================================================

def build_memory_context(
    session_id: str
):

    conversation = memory.format_memory(
        session_id
    )

    if not conversation:
        return "No previous conversation."

    return conversation


# ============================================================
# BUILD FINAL PROMPT
# ============================================================

def build_final_prompt(
    user_question: str,
    context: str,
    previous_memory: str
):

    formatted_prompt = prompt_template.format(

        retrieved_chunks=context,

        user_question=user_question
    )

    memory_instruction = f"""

============================================================
PREVIOUS CONVERSATION
============================================================

{previous_memory}

============================================================
MEMORY INSTRUCTIONS
============================================================

Use the previous conversation only to understand references
such as:

- he
- him
- his
- it
- that
- this
- there
- previous question
- previous answer

Do NOT treat conversation memory as a replacement for
Bharat's factual knowledge.

For factual information about Bharat, prefer the retrieved
knowledge above.

If the current question refers to something from the previous
conversation, use the conversation to understand what the
user means, then answer using the retrieved Bharat knowledge.

============================================================
END PREVIOUS CONVERSATION
============================================================
"""

    return (
        formatted_prompt
        + memory_instruction
    )


# ============================================================
# GEMINI STREAM
# ============================================================

def stream_gemini(
    prompt: str
) -> Generator[str, None, None]:

    client = get_gemini_client()

    print(
        f"Streaming from Gemini: {GEMINI_MODEL}"
    )

    response = client.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=prompt
    )

    for chunk in response:

        text = getattr(
            chunk,
            "text",
            None
        )

        if text:
            yield text


# ============================================================
# GROQ STREAM
# ============================================================

def stream_groq(
    prompt: str
) -> Generator[str, None, None]:

    client = get_groq_client()

    print(
        f"Streaming from Groq: {GROQ_MODEL}"
    )

    stream = client.chat.completions.create(

        model=GROQ_MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.3,

        stream=True
    )

    for chunk in stream:

        try:

            text = (
                chunk
                .choices[0]
                .delta
                .content
            )

            if text:
                yield text

        except Exception:
            continue


# ============================================================
# STREAM AI
# ============================================================

def stream_generate_answer(
    prompt: str
):

    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    try:

        got_token = False

        for text in stream_gemini(prompt):

            got_token = True

            yield text

        if got_token:
            return

    except Exception as gemini_error:

        print(
            f"Gemini streaming error: {gemini_error}"
        )

        if not is_quota_error(
            gemini_error
        ):

            raise gemini_error

        print(
            "Gemini quota exceeded."
        )

        print(
            "Switching to Groq streaming..."
        )

    # --------------------------------------------------------
    # GROQ
    # --------------------------------------------------------

    try:

        for text in stream_groq(prompt):

            yield text

    except Exception as groq_error:

        print(
            f"Groq streaming error: {groq_error}"
        )

        raise RuntimeError(
            "Both Gemini and Groq are currently unavailable."
        )


# ============================================================
# STREAM ANSWER
# ============================================================

def stream_answer(
    user_question: str,
    session_id: str = "default"
):

    # ========================================================
    # VALIDATION
    # ========================================================

    if not user_question:

        answer = (
            "I'm Jojo. What would you like to know?"
        )

        memory.add_exchange(
            session_id,
            user_question,
            answer
        )

        yield answer

        return

    user_question = user_question.strip()

    if not user_question:

        answer = (
            "I'm Jojo. What would you like to know?"
        )

        memory.add_exchange(
            session_id,
            user_question,
            answer
        )

        yield answer

        return


    # ========================================================
    # DIRECT JOJO RESPONSE
    # ========================================================

    direct_response = get_jojo_direct_response(
        user_question
    )

    if direct_response is not None:

        print(
            "Using Jojo direct response."
        )

        memory.add_exchange(
            session_id,
            user_question,
            direct_response
        )

        # Stream direct response too
        words = direct_response.split()

        for index, word in enumerate(words):

            if index > 0:
                yield " "

            yield word

        return


    # ========================================================
    # MEMORY
    # ========================================================

    previous_memory = build_memory_context(
        session_id
    )

    print()
    print("=" * 60)
    print(
        f"SESSION: {session_id}"
    )
    print("=" * 60)

    print(
        "Previous conversation:"
    )

    print(
        previous_memory
    )


    # ========================================================
    # DATABASE
    # ========================================================

    if not os.path.exists(DB_DIR):

        answer = (
            "System error: Vector database "
            "not initialized."
        )

        memory.add_exchange(
            session_id,
            user_question,
            answer
        )

        yield answer

        return


    try:

        # ====================================================
        # VECTOR STORE
        # ====================================================

        vector_store = get_vector_store()


        # ====================================================
        # SEARCH
        # ====================================================

        print()
        print("=" * 60)

        print(
            f"RAG QUESTION: {user_question}"
        )

        print(
            f"Retrieving top {TOP_K} chunks..."
        )

        print("=" * 60)

        results = (
            vector_store
            .similarity_search_with_score(
                user_question,
                k=TOP_K
            )
        )


        # ====================================================
        # NO RESULTS
        # ====================================================

        if not results:

            answer = (
                "I don't have that information "
                "about Bharat yet."
            )

            memory.add_exchange(
                session_id,
                user_question,
                answer
            )

            yield answer

            return


        # ====================================================
        # PRINT SOURCES
        # ====================================================

        print()
        print(
            "RETRIEVED SOURCES:"
        )

        for index, (doc, score) in enumerate(
            results,
            start=1
        ):

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            print(
                f"{index}. "
                f"{source} | "
                f"score={score:.4f}"
            )


        # ====================================================
        # BEST SCORE
        # ====================================================

        best_score = results[0][1]

        print()

        print(
            f"Best RAG score: "
            f"{best_score:.4f}"
        )


        # ====================================================
        # RELEVANCE
        # ====================================================

        if best_score > THRESHOLD:

            answer = (
                "I don't have that information "
                "about Bharat yet."
            )

            memory.add_exchange(
                session_id,
                user_question,
                answer
            )

            yield answer

            return


        # ====================================================
        # BUILD CONTEXT
        # ====================================================

        context = build_context(
            results
        )

        print(
            f"Context contains "
            f"{len(results)} chunks."
        )


        # ====================================================
        # BUILD PROMPT
        # ====================================================

        final_prompt = build_final_prompt(

            user_question=user_question,

            context=context,

            previous_memory=previous_memory
        )


        # ====================================================
        # STREAM AI
        # ====================================================

        full_answer = ""

        for chunk in stream_generate_answer(
            final_prompt
        ):

            if not chunk:
                continue

            full_answer += chunk

            # IMPORTANT:
            # send token immediately
            yield chunk


        # ====================================================
        # SAVE MEMORY ONLY AFTER COMPLETE RESPONSE
        # ====================================================

        full_answer = full_answer.strip()

        if full_answer:

            memory.add_exchange(
                session_id,
                user_question,
                full_answer
            )


    except Exception as e:

        print()
        print(
            f"Unexpected streaming error: {e}"
        )

        error_message = (
            "I'm having trouble processing "
            "that request right now."
        )

        yield error_message


# ============================================================
# NON-STREAMING COMPATIBILITY
# ============================================================

def answer_question(
    user_question: str,
    session_id: str = "default"
):

    chunks = []

    for chunk in stream_answer(
        user_question,
        session_id
    ):

        chunks.append(
            chunk
        )

    return "".join(
        chunks
    )