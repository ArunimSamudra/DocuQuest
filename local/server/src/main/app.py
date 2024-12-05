from contextlib import asynccontextmanager
from typing import Dict

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel

from main.config import Config
from main.modules.doc_handler import extract_text_from_pdf
from main.request_handler import RequestHandler
from mlx_lm import load

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    model, tokenizer = load(Config.LOCAL_MODEL_PATH)
    print("Model loaded successfully.")
    ml_models['model'] = model
    ml_models['tokenizer'] = tokenizer
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()


app = FastAPI(lifespan=lifespan)
request_handler = RequestHandler()
# In-memory session storage
session_store: Dict[str, Dict] = {}


@app.post("/summarize")
async def summarize_pdf(
        session_id: str = Form(...),
        file: UploadFile = File(None),
        text: str = Form(None),  # For receiving text directly
        file_type: str = Form(None)  # For receiving file_type directly from form
):
    if session_id not in session_store:
        session_store[session_id] = {}
    # If file is uploaded
    if file and file_type == 'pdf':
        pdf_content = await file.read()
        document_text = extract_text_from_pdf(pdf_content)
    # If text content is provided
    elif text and file_type == "txt":
        document_text = text
    else:
        raise HTTPException(status_code=400, detail="No file or text content provided.")
    response = await request_handler.summarize_text(ml_models['model'], ml_models['tokenizer'], session_id,
                                                    document_text)
    return response


class QuestionRequest(BaseModel):
    question: str
    session_id: str


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    if request.session_id not in session_store:
        raise HTTPException(status_code=400, detail="Session not found.")
    return await request_handler.answer(ml_models['model'], ml_models['tokenizer'], request.session_id,
                                        request.question)


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Deletes the session's vector store and other related data.
    """
    await request_handler.delete_session(session_id)


@app.get("/hello")
async def hello_world():
    return "Hello World!"


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
