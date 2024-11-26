import requests
import os
from dotenv import load_dotenv

load_dotenv()


def get_models():
    api_key = os.environ.get("GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1/models"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        response = requests.get(url, headers=headers)
        data = response.json()["data"]
        models = []
        for model in data:
            models.append(model.get("id"))
        return models
    except Exception as e:
        return ["Error: Can't recover models from Groq API", e]


def process_model_json(response):
    data = response["data"]
    models = []
    for model in data:
        models.append(model.get("id"))

    return models


if __name__ == "__main__":
    print(get_models())
    # print(process_model_json(get_models()))
