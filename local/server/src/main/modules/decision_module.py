import pickle
import textstat
import psutil

from main.config import Config


def use_cloud_llm(document_text):
    # Route to cloud if the document is large or complex
    # return len(document_text.split()) > Config.DECISION_THRESHOLD
    # returning True always for now
    # 4. Maybe a feedback mechanism

    # Step 2: Check document length
    # word_count = len(document_text.split())
    # if word_count > Config.DECISION_THRESHOLD:
    #     print(f"Routing to cloud: Document is too large ({word_count} words).")
    #     return True

    # Step 1: Check local system health
    # cpu_usage, memory_available, disk_free = get_local_system_metrics()
    # # Decision logic
    # if cpu_usage > Config.CPU_THRESHOLD:
    #     print(f"Routing to cloud: CPU usage too high ({cpu_usage}%).")
    #     return True
    # if memory_available < Config.MEMORY_THRESHOLD:
    #     print(f"Routing to cloud: Insufficient memory available ({memory_available} MB).")
    #     return True
    # if disk_free < Config.DISK_THRESHOLD:
    #     print(f"Routing to cloud: Insufficient disk space ({disk_free} GB).")
    #     return True
    #
    # # Step 3: Predict document complexity using the classifier
    # prediction = evaluate_document_complexity(document_text)
    #
    # # Decide based on the prediction
    # if prediction == 0: ##easy
    #     print("Processing locally: Document is classified as easy.")
    #     return False
    # elif prediction == 1: ##medium
    #     print("Routing to cloud: Document is classified as medium complexity.")
    #     return True
    # elif prediction == 2: ##diffcult
    #     print("Routing to cloud: Document is classified as difficult.")
    #     return True
    #
    # # Default fallback
    # print("Processing locally: Default fallback.")
    return False

def get_local_system_metrics():
    """
    Retrieve local system metrics such as CPU usage, memory availability, and disk usage.

    Returns:
        dict: A dictionary containing CPU usage, available memory, and disk usage.
    """
    # CPU Usage
    cpu_usage = psutil.cpu_percent(interval=1)

    # Memory Info
    memory_info = psutil.virtual_memory()
    memory_available = memory_info.available / (1024**2)  # MB

    # Disk Info
    disk_info = psutil.disk_usage('/')
    disk_free = disk_info.free / (1024**3)  # GB

    return cpu_usage, memory_available, disk_free

# Example Usage
metrics = get_local_system_metrics()
print(metrics)


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

    feature_vector = [
        avg_word_length,
        type_token_ratio,
        avg_sentence_length,
        fk_score
    ]

    with open(Config.CLASSIFIER_MODEL_PATH, 'rb') as file:
        classifier = pickle.load(file)
    prediction = classifier.predict([feature_vector])[0]  # Predict complexity level

    return prediction