from main.config import Config
from main.impl.llm import LLM
import requests
import json


class Cloud(LLM):

    def summarize_text(self, text):
        prompt = (
            "You are a highly intelligent and concise text summarization assistant. "
            "Your task is to summarize the given text clearly and accurately. "
            "Keep the summary informative yet brief. Do not output your thought process, just output the summary\n\n"
            f"Text to be summarized: {text}"
        )
        payload = json.dumps({
            "prompt": prompt
        })
        headers = {
            'Connection': 'keep-alive',
            'Content-Type': 'application/json',
            'Authorization': 'Bearer 0lXKBW3n6nYYderfIrxE-JeIXjDxKpKBRvOC7Z2anpRuPj_ZTLg8_DJhcCk9-bckksSpWSr1bzf0z_Nw86Klbw=='
        }

        try:
            response = requests.post(Config.CLOUD_API_URL, headers=headers, data=payload, timeout=30)
            return json.loads(response.text)
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
        payload = json.dumps({
            "prompt": prompt
        })
        headers = {
            'Connection': 'keep-alive',
            'Content-Type': 'application/json',
            'Authorization': 'Bearer 0lXKBW3n6nYYderfIrxE-JeIXjDxKpKBRvOC7Z2anpRuPj_ZTLg8_DJhcCk9-bckksSpWSr1bzf0z_Nw86Klbw=='
        }

        try:
            response = requests.post(Config.CLOUD_API_URL, headers=headers, data=payload, timeout=30)
            return json.loads(response.text)
        except requests.exceptions.Timeout:
            return {"error": "The request timed out after 30 seconds."}
        except requests.exceptions.RequestException as e:
            return {"error": f"An error occurred: {str(e)}"}