import os

from main.config import Config
from main.impl.llm import LLM
import json
import requests
import google.auth
import google.auth.transport.requests
from google.oauth2 import service_account


class Cloud(LLM):

    def __init__(self):
        SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
        src_directory = os.path.abspath(os.path.join(Config.ROOT_DIR, '..'))
        SERVICE_ACCOUNT_FILE = os.path.join(src_directory, 'resources', Config.SERVICE_ACCOUNT_FILE_NAME)
        cred = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        auth_req = google.auth.transport.requests.Request()
        cred.refresh(auth_req)
        self.bearer_token = cred.token
        project_id = "198585961536"
        endpoint_id = "3797775834501087232"
        base_url = "https://us-central1-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-central1/endpoints/{endpoint_id}:predict"
        self.full_url = base_url.format(project_id=project_id, endpoint_id=endpoint_id)

    def summarize_text(self, text):
        request_body = {
            "instances": [
                {
                    "prompt": "You are a highly intelligent and concise text summarization assistant. "
                              "Your task is to extract the key points and summarize the given text clearly and accurately. "
                              "Keep the summary informative yet brief."
                              f"Text to be summarized: {text}",
                    "max_tokens": 1000,
                    "top_k": 1,
                    "top_p": 1.0,
                    "temperature": 0.2
                }
            ]
        }
        headers = {
            "Authorization": "Bearer {bearer_token}".format(bearer_token=self.bearer_token),
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.full_url, json=request_body, headers=headers, timeout=30)
            response = response.json()
            predictions = response["predictions"]
            output_text = predictions[0]
            output_text = output_text.split("Output:", 1)[-1].strip()
            return json.loads(output_text)
        except requests.exceptions.Timeout:
            return {"error": "The request timed out after 30 seconds."}
        except requests.exceptions.RequestException as e:
            return {"error": f"An error occurred: {str(e)}"}

    def answer(self, context, question):
        prompt = (
            "You are a highly intelligent and context-aware assistant. Your role is to help the user "
            "answer questions accurately and concisely based on the provided context. Use the context "
            "carefully to generate relevant responses. If the context does not contain enough information "
            "to answer the question, let the user know by stating: 'The context does not provide enough information to answer this question.'"
            f"\n\nContext: {context}\n\nQuestion: {question}"
        )

        request_body = {
            "instances": [
                {
                    "prompt": prompt,
                    "max_tokens": 1000,
                    "top_k": 1,
                    "top_p": 1.0,
                    "temperature": 0.2
                }
            ]
        }

        headers = {
            "Authorization": "Bearer {bearer_token}".format(bearer_token=self.bearer_token),
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.full_url, json=request_body, headers=headers, timeout=30)
            response = response.json()
            predictions = response["predictions"]
            output_text = predictions[0]
            output_text = output_text.split("Output:", 1)[-1].strip()
            return json.loads(output_text)
        except requests.exceptions.Timeout:
            return {"error": "The request timed out after 30 seconds."}
        except requests.exceptions.RequestException as e:
            return {"error": f"An error occurred: {str(e)}"}
