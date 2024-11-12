from src.main.config import DECISION_THRESHOLD


def use_cloud_llm(document_text):
    # Route to cloud if the document is large or complex
    return len(document_text.split()) > DECISION_THRESHOLD
