import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = "product-manual" # Choose a name for your Pinecone index
PDF_PATH = "data/Samsung-user-manual.pdf"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # A powerful, open-source embedding model

def run_ingestion():
    """
    Loads data from PDF, splits it, creates embeddings, and stores in Pinecone.
    """
    if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT]):
        print("Error: Pinecone API Key or Environment not found in .env file.")
        return

    print("Step 1: Loading document...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    if not documents:
        print(f"Error: Could not load document from {PDF_PATH}")
        return
    print(f"Successfully loaded {len(documents)} pages.")

    print("\nStep 2: Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"Document split into {len(docs)} chunks.")

    print("\nStep 3: Initializing embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print("Embeddings model initialized.")

    print(f"\nStep 4: Upserting documents to Pinecone index '{PINECONE_INDEX_NAME}'...")
    # This will create a new index if it doesn't exist or use an existing one.
    PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )
    print("✅ Ingestion complete! Your knowledge base is ready.")


if __name__ == "__main__":
    run_ingestion()