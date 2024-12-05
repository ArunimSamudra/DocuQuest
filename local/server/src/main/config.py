import os


class Config:
    SERVER_URL = "http://127.0.0.1:8080"
    DECISION_THRESHOLD = 1000  # Word limit for using cloud LLM
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    LOCAL_MODEL_PATH = os.path.join(ROOT_DIR, "models/llm")
    CLASSIFIER_MODEL_PATH = os.path.join(ROOT_DIR, "models/classifier/logistic_regression_model.pkl")
    CLOUD_API_URL = 'https://cloud-6d541cf-v4.app.beam.cloud'
    CPU_THRESHOLD = 70
    MEMORY_THRESHOLD = 3000  # MB
    DISK_THRESHOLD = 10  # GB for LLaMA 3.2 3B