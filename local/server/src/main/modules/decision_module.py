import textstat

def use_cloud_llm(document_text):
    # Route to cloud if the document is large or complex
    # return len(document_text.split()) > Config.DECISION_THRESHOLD
    # returning True always for now
    # 1. Check local system health
    # 2. Check doc length
    # 3. Evaluate document complexity
    # 4. Maybe a feedback mechanism
    return False

def evaluate_document_complexity(text):
    """
    Calculate a simple complexity score based on lexical and structural features.

    Args:
        text (str): The document text.

    Returns:
        int: A score representing the document's complexity.
    """
    words = text.split()
    sentences = text.split('.')

    # Lexical features
    avg_word_length = sum(len(word) for word in words) / len(words)
    type_token_ratio = len(set(words)) / len(words)

    # Sentence features
    avg_sentence_length = len(words) / len(sentences)

    # Flesch-Kincaid Readability Score (lower score indicates higher complexity)
    fk_score = textstat.flesch_reading_ease(text)
