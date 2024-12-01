def use_cloud_llm(document_text):
    # Route to cloud if the document is large or complex
    # return len(document_text.split()) > Config.DECISION_THRESHOLD
    # returning True always for now
    return False
