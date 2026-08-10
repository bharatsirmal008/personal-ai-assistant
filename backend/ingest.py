import os
import pdfplumber
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

PDF_PATH = "../profile/public/Bharat_Sirmal_Resume.pdf"
DB_DIR = "./chroma_db"

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def chunk_text(text):
    # Using RecursiveCharacterTextSplitter for logical chunking,
    # tuning chunk size to be small enough to capture sections well.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    return documents

def ingest_data():
    print(f"Reading PDF from {PDF_PATH}...")
    text = extract_text_from_pdf(PDF_PATH)
    if not text:
        print("No text extracted, aborting ingestion.")
        return

    print("Chunking text...")
    documents = chunk_text(text)
    print(f"Generated {len(documents)} chunks.")

    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Storing in ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    vector_store.persist()
    print(f"Successfully ingested data into {DB_DIR}")

if __name__ == "__main__":
    ingest_data()
