import os
from openai import OpenAI
from .Model import Model


def _resolve_key(value):
    """Resolve ENV:<VAR_NAME> to the actual environment variable value."""
    if isinstance(value, str) and value.startswith("ENV:"):
        env_var = value[4:]
        resolved = os.environ.get(env_var)
        if not resolved:
            raise ValueError(f"Environment variable '{env_var}' is not set. Export it before running.")
        return resolved
    return value


class GPT(Model):
    def __init__(self, config):
        super().__init__(config)
        api_keys = config["api_key_info"]["api_keys"]
        api_pos = int(config["api_key_info"]["api_key_use"])
        assert (0 <= api_pos < len(api_keys)), "Please enter a valid API key to use"
        self.max_output_tokens = int(config["params"]["max_output_tokens"])

        api_key = _resolve_key(api_keys[api_pos])
        model_info = config.get("model_info", {})
        base_url = model_info.get("base_url", None)
        default_headers = model_info.get("default_headers", None)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
        )

    def query(self, msg):
        try:
            completion = self.client.chat.completions.create(
                model=self.name,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg}
                ],
            )
            response = completion.choices[0].message.content
           
        except Exception as e:
            print(e)
            response = ""

        return response