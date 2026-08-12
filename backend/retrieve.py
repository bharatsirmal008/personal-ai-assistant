import os

from dotenv import load_dotenv
from google import genai
from groq import Groq

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from prompt import prompt_template


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ENV_FILE = os.path.join(BASE_DIR, ".env")

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

# Lower L2 distance = more relevant
THRESHOLD = 1.7

# Number of documents retrieved from ChromaDB
TOP_K = 2


# ============================================================
# GEMINI SETTINGS
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash"
)


# ============================================================
# GROQ SETTINGS
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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
    """
    Create the Gemini client once and reuse it.
    """

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
    """
    Create the Groq client once and reuse it.
    """

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
# LOAD VECTOR DATABASE
# ============================================================

def get_vector_store():
    """
    Lazily initialize the embedding model and ChromaDB.
    """

    global _embeddings
    global _vector_store

    if _vector_store is None:

        print(
            "Loading embedding model and vector store..."
        )

        # ----------------------------------------------------
        # Embedding Model
        # ----------------------------------------------------

        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # ----------------------------------------------------
        # ChromaDB
        # ----------------------------------------------------

        _vector_store = Chroma(
            persist_directory=DB_DIR,
            embedding_function=_embeddings
        )

        print(
            "Embedding model and Chroma DB loaded."
        )

    return _vector_store


# ============================================================
# CALL GEMINI
# ============================================================

def call_gemini(prompt: str) -> str:
    """
    Send the final RAG prompt to Gemini.
    """

    client = get_gemini_client()

    print(
        f"Sending request to Gemini: {GEMINI_MODEL}"
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return response.text.strip()


# ============================================================
# CALL GROQ
# ============================================================

def call_groq(prompt: str) -> str:
    """
    Send the final RAG prompt to Groq.
    """

    client = get_groq_client()

    print(
        f"Sending request to Groq: {GROQ_MODEL}"
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
    )

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError(
            "Groq returned an empty response."
        )

    return answer.strip()


# ============================================================
# CHECK GEMINI QUOTA ERROR
# ============================================================

def is_quota_error(error: Exception) -> bool:
    """
    Detect Gemini quota/rate-limit errors.
    """

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
# GENERATE ANSWER WITH AUTOMATIC FALLBACK
# ============================================================

def generate_answer(prompt: str) -> str:
    """
    AI model router:

    1. Try Gemini
    2. If Gemini quota is exhausted, use Groq
    3. If both fail, return an error
    """

    # ========================================================
    # TRY GEMINI FIRST
    # ========================================================

    try:

        return call_gemini(prompt)

    except Exception as gemini_error:

        print(
            f"Gemini error: {gemini_error}"
        )

        # ----------------------------------------------------
        # Only fallback for quota/rate-limit errors
        # ----------------------------------------------------

        if not is_quota_error(gemini_error):

            raise gemini_error

        print(
            "Gemini quota exceeded."
        )

        print(
            "Switching automatically to Groq..."
        )


    # ========================================================
    # FALLBACK TO GROQ
    # ========================================================

    try:

        return call_groq(prompt)

    except Exception as groq_error:

        print(
            f"Groq error: {groq_error}"
        )

        raise RuntimeError(
            "Both Gemini and Groq are currently unavailable."
        )


# ============================================================
# JOJO DIRECT RESPONSES
# ============================================================

def get_jojo_direct_response(user_question: str):
    """
    Handle simple Jojo identity and greeting questions
    without using ChromaDB or Gemini.

    This makes common questions much faster.
    """

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
        return "Hi! I'm Jojo. How can I help you?"

    # --------------------------------------------------------
    # Jojo's name
    # --------------------------------------------------------

    if q in [
        "what is your name",
        "what's your name",
        "what is ur name",
        "your name",
        "tell me your name"
    ]:
        return "My name is Jojo."

    # --------------------------------------------------------
    # Who is Jojo?
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
    # Jojo's boss
    # --------------------------------------------------------

    if q in [
        "who is your boss",
        "who's your boss",
        "who is ur boss",
        "who is your owner",
        "who owns you"
    ]:
        return "Bharat Sirmal is my boss."

    # --------------------------------------------------------
    # Not a direct question
    # --------------------------------------------------------

    return None


# ============================================================
# ANSWER QUESTION
# ============================================================

def answer_question(user_question: str) -> str:
    """
    Main RAG pipeline:

    1. Check simple Jojo questions
    2. If simple question, answer immediately
    3. Otherwise load ChromaDB
    4. Search relevant documents
    5. Check relevance score
    6. Build context
    7. Create final prompt
    8. Try Gemini
    9. If Gemini quota is exhausted, use Groq
    10. Return answer
    """

    # ========================================================
    # VALIDATE QUESTION
    # ========================================================

    if not user_question or not user_question.strip():

        return "I'm Jojo. What would you like to know?"


    # ========================================================
    # JOJO DIRECT RESPONSE
    # ========================================================

    direct_response = get_jojo_direct_response(
        user_question
    )

    if direct_response is not None:

        print(
            "Using Jojo direct response."
        )

        return direct_response


    # ========================================================
    # CHECK CHROMA DATABASE
    # ========================================================

    if not os.path.exists(DB_DIR):

        return (
            "System error: Vector database "
            "not initialized."
        )


    try:

        # ====================================================
        # LOAD VECTOR STORE
        # ====================================================

        vector_store = get_vector_store()


        # ====================================================
        # SEARCH CHROMA DATABASE
        # ====================================================

        print(
            f"Searching ChromaDB for: {user_question}"
        )

        results = vector_store.similarity_search_with_score(
            user_question,
            k=TOP_K
        )


        # ====================================================
        # NO RESULTS
        # ====================================================

        if not results:

            return (
                "I don't have that information about "
                "Bharat yet."
            )


        # ====================================================
        # BEST SCORE
        # ====================================================

        best_score = results[0][1]

        print(
            f"Best RAG score: {best_score}"
        )


        # ====================================================
        # CHECK RELEVANCE
        # ====================================================

        if best_score > THRESHOLD:

            return (
                "I don't have that information about "
                "Bharat yet."
            )


        # ====================================================
        # BUILD CONTEXT
        # ====================================================

        context = "\n\n".join(
            [
                doc.page_content
                for doc, score in results
            ]
        )


        # ====================================================
        # CREATE PROMPT
        # ====================================================

        formatted_prompt = prompt_template.format(
            retrieved_chunks=context,
            user_question=user_question
        )


        # ====================================================
        # AI MODEL ROUTER
        # ====================================================

        return generate_answer(
            formatted_prompt
        )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        print(
            f"Unexpected RAG error: {e}"
        )

        return (
            f"Unexpected error: {str(e)}"
        )