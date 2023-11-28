import os
from typing import Dict, Any
from haystack.nodes import PromptNode

def createPromptNode(generative_model, prompt_template, max_length):
    
    models_config: Dict[str, Any] = {
        "mistral": {"model_name": "mistralai/Mistral-7B-Instruct-v0.1",
                    "api_key": os.getenv('HUGGINGFACEHUB_API_TOKEN')},
        "falcon": {"model_name": "tiiuae/falcon-7b-instruct",
                    "api_key": os.getenv('HUGGINGFACEHUB_API_TOKEN')},
        "gpt-3.5-turbo": {"model_name": "gpt-3.5-turbo",
                    "api_key": os.getenv('OPENAI_API_KEY')},
        "gpt-4": {"model_name": "gpt-4",
                    "api_key": os.getenv('OPENAI_API_KEY')},
        "gpt-4-turbo": {"model_name": "gpt-4-1106-preview",
                    "api_key": os.getenv('OPENAI_API_KEY')},
        "command": {"model_name": "command",
                    "api_key": os.getenv('COHERE_API_KEY')},
    }

    model: Dict[str, str] = models_config[generative_model]
    prompt_node = PromptNode(model_name_or_path=model["model_name"],
                             api_key=model["api_key"],
                             default_prompt_template=prompt_template,
                             max_length=max_length)

    return prompt_node