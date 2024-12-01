class Config:
    SERVER_URL = "http://127.0.0.1:8080"
    DECISION_THRESHOLD = 1000  # Word limit for using cloud LLM
    LOCAL_MODEL_PATH = "server/src/main/models/model"
    CLOUD_API_URL = "https://cloud-llm-endpoint.com/api"