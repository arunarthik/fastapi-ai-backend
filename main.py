import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import httpx
from rag import load_resume_into_vector_db, search_resume


RESUME_URL = "https://raw.githubusercontent.com/arunarthik/arun-ai-backend/main/data/resume.json"
RESUME_DATA = {}


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class ChatRequest(BaseModel):
    message: str

async def load_resume():
    global RESUME_DATA
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.get(RESUME_URL)
            RESUME_DATA = res.json()
            print("✅ Resume loaded from GitHub")
    except Exception as e:
        print("❌ Failed to load resume:", str(e))
@app.on_event("startup")
async def startup_event():
    await load_resume()
    load_resume_into_vector_db(RESUME_DATA)




@app.get("/")
async def root():
    return {"status": "FastAPI AI backend running"}


@app.post("/chat")
async def chat(req: ChatRequest):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="Missing OPENROUTER_API_KEY")

    relevant_chunks = search_resume(req.message)
    context = "\n".join(relevant_chunks)
    
    payload = {
        "model": "openai/gpt-3.5-turbo",  # free & reliable
        "messages": [
            {
            "role": "system",
            "content": (
                "You are an AI assistant that answers ONLY based on the following resume data.\n"
                "If the answer is not in the resume, say you don't know.\n\n"
        f"RESUME DATA:\n{json.dumps(RESUME_DATA, indent=2)}"
    ),
},

            {"role": "user", "content": req.message},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            res = await client.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

        if res.status_code != 200:
            print("OPENROUTER HTTP ERROR:", res.text)
            return {"reply": f"AI HTTP error: {res.status_code}"}

        data = res.json()

        reply = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "No response from AI.")
        )

        return {"reply": reply}

    except Exception as e:
        print("OpenRouter error:", str(e))
        return {"reply": "Something went wrong. Please try again in a moment."}

