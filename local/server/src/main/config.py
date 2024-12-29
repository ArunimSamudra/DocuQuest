import os


class Config:
    SERVER_URL = "http://127.0.0.1:8080"
    # Insert your openai api key here
    OPENAI_API_KEY = ''
    DECISION_THRESHOLD = 5000  # Word limit for using local LLM
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    SERVICE_ACCOUNT_FILE_NAME = 'personal-assistant-440203-a23b1a7ea851.json'
    LOCAL_MODEL_PATH = os.path.join(ROOT_DIR, "models/llm")
    CLASSIFIER_MODEL_PATH = os.path.join(ROOT_DIR, "models/classifier/logistic_regression_model.pkl")
    CPU_THRESHOLD = 70
    MEMORY_THRESHOLD = 3000  # MB