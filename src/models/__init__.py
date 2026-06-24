from .GPT import GPT
import json


def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results


def create_model(config_path):
    config = load_json(config_path)
    provider = config["model_info"]["provider"].lower()

    if provider == 'gpt':
        return GPT(config)
    else:
        raise ValueError(f"ERROR: Unknown provider '{provider}'. Only 'gpt' (OpenRouter/OpenAI) is supported.")
