import chromadb
import ollama

# Connect to existing ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("langchain")

# Ask the user
question = input("You: ")

# Search your personal data
results = collection.query(
    query_texts=[question],
    n_results=3
)

# Build context
context = "\n\n".join(results["documents"][0])

# Send retrieved information to local Llama
prompt = f"""
You are Bharat Sirmal's personal AI assistant.

Answer the user's question using the personal information provided below.

PERSONAL INFORMATION:
{context}

USER QUESTION:
{question}

Rules:
- Answer using the provided information.
- Do not invent personal information.
- If the information is not available, say:
  "I don't have that information yet."
"""

response = ollama.chat(
    model="llama3.2:latest",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nAI:", response["message"]["content"])