import os
import requests
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from prompt import prompt_template

DB_DIR = "./chroma_db"
# You may adjust this threshold empirically for all-MiniLM-L6-v2
THRESHOLD = 1.7  # Increased threshold for L2 distance with all-MiniLM-L6-v2

# --- Loaded once and reused across all requests (this was the main slowdown) ---
_embeddings = None
_vector_store = None


def get_vector_store():
    """Lazily initialize the embeddings model and Chroma DB exactly once."""
    global _embeddings, _vector_store
    if _vector_store is None:
        print("Loading embedding model and vector store (one-time)...")
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _vector_store = Chroma(persist_directory=DB_DIR, embedding_function=_embeddings)
    return _vector_store


def call_local_llm(prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 80},
            "keep_alive": "30m",  # keep the model resident in memory between requests
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def answer_question(user_question: str) -> str:
    if not os.path.exists(DB_DIR):
        return "System error: Vector database not initialized."

    vector_store = get_vector_store()

    results = vector_store.similarity_search_with_score(user_question, k=3)

    if not results:
        return "I can only answer questions about Bharat Sirmal's background and experience."

    best_score = results[0][1]
    print(f"Best score: {best_score}")

    # In Chroma default L2, a lower score is better.
    # If the score is higher than our threshold (distance is too far), reject.
    if best_score > THRESHOLD:
        return "I can only answer questions about Bharat Sirmal's background and experience."

    context = "\n\n".join([doc.page_content for doc, score in results])

    formatted_prompt = prompt_template.format(
        retrieved_chunks=context,
        user_question=user_question
    )

    try:
        return call_local_llm(formatted_prompt)
    except requests.exceptions.RequestException as e:
        return f"Error connecting to local LLM: {str(e)}. Make sure Ollama is running."
    except Exception as e:
        return f"Unexpected error: {str(e)}"