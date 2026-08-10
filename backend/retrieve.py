import os
import requests

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from prompt import prompt_template


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_DIR = os.path.join(
    BASE_DIR,
    "chroma_db"
)


# ============================================================
# RAG SETTINGS
# ============================================================

THRESHOLD = 1.7


# ============================================================
# OLLAMA SETTINGS
# ============================================================

# Railway:
# OLLAMA_URL =
# http://ollama.railway.internal:11434/api/generate
#
# Local computer:
# http://127.0.0.1:11434/api/generate

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434/api/generate"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2"
)


# ============================================================
# GLOBAL CACHE
# ============================================================

_embeddings = None
_vector_store = None


# ============================================================
# LOAD VECTOR DATABASE
# ============================================================

def get_vector_store():
    """
    Lazily initialize the embedding model and Chroma DB
    exactly once.
    """

    global _embeddings
    global _vector_store

    if _vector_store is None:

        print(
            "Loading embedding model and vector store..."
        )

        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        _vector_store = Chroma(
            persist_directory=DB_DIR,
            embedding_function=_embeddings
        )

        print(
            "Embedding model and Chroma DB loaded."
        )

    return _vector_store


# ============================================================
# CALL OLLAMA
# ============================================================

def call_local_llm(prompt: str) -> str:
    """
    Send the prompt to Ollama.

    Local:
        http://127.0.0.1:11434/api/generate

    Railway:
        http://ollama.railway.internal:11434/api/generate
    """

    print(
        f"Sending request to Ollama: {OLLAMA_URL}"
    )

    response = requests.post(

        OLLAMA_URL,

        json={

            "model": OLLAMA_MODEL,

            "prompt": prompt,

            "stream": False,

            "options": {
                "temperature": 0.3,
                "num_predict": 80
            },

            "keep_alive": "30m"
        },

        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    if "response" not in data:

        raise RuntimeError(
            f"Unexpected Ollama response: {data}"
        )

    return data["response"].strip()


# ============================================================
# ANSWER QUESTION
# ============================================================

def answer_question(user_question: str) -> str:

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
        # SEARCH
        # ====================================================

        results = (
            vector_store
            .similarity_search_with_score(
                user_question,
                k=3
            )
        )

        # ====================================================
        # NO RESULTS
        # ====================================================

        if not results:

            return (
                "I can only answer questions about "
                "Bharat Sirmal's background and experience."
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

        # Chroma default L2 distance:
        # lower = better
        #
        # If distance is greater than threshold,
        # the question is probably unrelated.

        if best_score > THRESHOLD:

            return (
                "I can only answer questions about "
                "Bharat Sirmal's background and experience."
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
        # SEND TO OLLAMA
        # ====================================================

        return call_local_llm(
            formatted_prompt
        )

    # ========================================================
    # REQUEST ERROR
    # ========================================================

    except requests.exceptions.ConnectionError:

        return (
            "Error connecting to Ollama. "
            "Please make sure the Ollama service "
            "is running."
        )

    except requests.exceptions.Timeout:

        return (
            "Ollama took too long to respond. "
            "Please try again."
        )

    except requests.exceptions.RequestException as e:

        return (
            f"Error connecting to Ollama: {str(e)}"
        )

    # ========================================================
    # OTHER ERROR
    # ========================================================

    except Exception as e:

        print(
            f"Unexpected RAG error: {e}"
        )

        return (
            f"Unexpected error: {str(e)}"
        )