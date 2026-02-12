import os
import chromadb
import httpx

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBED_URL = "https://openrouter.ai/api/v1/embeddings"

client = chromadb.Client()
collection = client.get_or_create_collection(name="resume")


async def get_embedding(text: str):
    async with httpx.AsyncClient(timeout=30) as http:
        res = await http.post(
            EMBED_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "text-embedding-3-small",
                "input": text,
            },
        )

    data = res.json()
    return data["data"][0]["embedding"]


async def load_resume_into_vector_db(resume_data: dict):
    docs = []

    for key, value in resume_data.items():
        if isinstance(value, list):
            for item in value:
                docs.append(f"{key}: {item}")
        else:
            docs.append(f"{key}: {value}")

    embeddings = []
    for doc in docs:
        emb = await get_embedding(doc)
        embeddings.append(emb)

    collection.add(
        documents=docs,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(docs))],
    )


async def search_resume(query: str, k: int = 3):
    query_embedding = await get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )

    return results["documents"][0]
