
from fastapi import FastAPI
from pydantic import BaseModel
from inference import predict_message
from typing import Any

app = FastAPI(title="PhishDetect API")

class MessageIn(BaseModel):
    text: str

class MessageOut(BaseModel):
    score: float
    label: str
    evidence: Any
    action: str

@app.post("/analyze", response_model=MessageOut)
def analyze(msg: MessageIn):
    return predict_message(msg.text)

@app.get("/health")
def health():
    return {"status": "ok"}
