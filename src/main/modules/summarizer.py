from decision_module import use_cloud_llm


def summarize_document(document_text):
    if use_cloud_llm(document_text):
        return cloud_summarize(document_text)
    return local_summarize(document_text)
