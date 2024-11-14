from fastapi import FastAPI, UploadFile, File, HTTPException
from src.main.modules.doc_handler import extract_text_from_pdf
from src.main.request_handler import RequestHandler

app = FastAPI()

@app.post("/summarize")
async def summarize_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file format.")
    pdf_content = await file.read()
    document_text = extract_text_from_pdf(pdf_content)
    summary = RequestHandler().summarize_text(document_text)
    return {"summary": summary}

@app.post("/ask")
async def ask_question(question: str):
    return "Sample Answer"
