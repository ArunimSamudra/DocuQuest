import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException, Form

from main.modules.doc_handler import extract_text_from_pdf

app = FastAPI()


class TextRequest(BaseModel):
    text: str
    file_type: str


@app.post("/summarize")
async def summarize_pdf(
    file: UploadFile = File(None),
    text: str = Form(None),  # For receiving text directly
    file_type: str = Form(None)  # For receiving file_type directly from form
):
    # If file is uploaded
    if file and file_type == 'pdf':
        pdf_content = await file.read()
        document_text = extract_text_from_pdf(pdf_content)
    # If text content is provided
    elif text and file_type == "txt":
        document_text = text
    else:
        raise HTTPException(status_code=400, detail="No file or text content provided.")

    return {"summary": document_text}


@app.post("/ask")
async def ask_question(question: str):
    return "Sample Answer"


@app.get("/hello")
async def hello_world():
    return "Hello World!"


if __name__ == "__main__":
    # export PYTHONPATH=$(pwd)/src
    uvicorn.run(app, host="127.0.0.1", port=8080)
