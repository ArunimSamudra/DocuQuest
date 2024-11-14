from fastapi import FastAPI, UploadFile, File, HTTPException

from main.modules.doc_handler import extract_text_from_pdf

app = FastAPI()

@app.post("/summarize")
async def summarize_pdf(file: UploadFile = File(...)):
    # Check the file type
    if file.content_type == "application/pdf":
        pdf_content = await file.read()
        document_text = extract_text_from_pdf(pdf_content)
    elif file.content_type == "text/plain":
        txt_content = await file.read()
        document_text = txt_content.decode("utf-8")
    else:
        raise HTTPException(status_code=400, detail="Invalid file format. Only PDF and TXT files are supported.")
    #summary = RequestHandler().summarize_text(document_text)
    return {"summary": document_text}

@app.post("/ask")
async def ask_question(question: str):
    return "Sample Answer"
