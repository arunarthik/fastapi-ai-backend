import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent"


class ChatRequest(BaseModel):
    message: str


@app.get("/")
async def root():
    return {"status": "FastAPI AI backend running"}


@app.post("/chat")
async def chat(req: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing GEMINI_API_KEY")

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "You are Arun's portfolio AI assistant. "
                            "Answer briefly and professionally.\n\n"
                            f"User: {req.message}"
                        )
                    }
                ],
            }
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            res = await client.post(
                f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                json=payload,
            )

        data = res.json()

        if "error" in data:
            print("Gemini API error:", data)
            return {"reply": "AI is temporarily unavailable. Please try again."}

        reply = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "No response from AI.")
        )

        return {"reply": reply}

    except Exception as e:
        print("Gemini error:", str(e))
        return {"reply": "Something went wrong. Please try again in a moment."}
