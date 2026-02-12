import json
import chromadb
from sentence_transformers import SentenceTransformer


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create in-memory Chroma DB
client = chromadb.Client()
collection = client.get_or_create_collection(name="resume")

def load_resume_into_vector_db(resume_data: dict):
    """
    Convert resume sections into vector embeddings
    and store in ChromaDB.
    """
    documents = []

    for key, value in resume_data.items():
        if isinstance(value, list):
            for item in value:
                documents.append(f"{key}: {item}")
        else:
            documents.append(f"{key}: {value}")

    embeddings = model.encode(documents).tolist()

    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(documents))],
    )


def search_resume(query: str, k: int = 3):
    """
    Return top-k relevant resume chunks.
    """
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
    )

    return results["documents"][0]
