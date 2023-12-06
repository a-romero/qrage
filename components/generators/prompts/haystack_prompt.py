import os
from typing import Dict, Any
from components.generators.models import Model
from haystack.nodes import PromptNode

def createPromptNode(
        generative_model, 
        prompt_template, 
        max_length
):
    """
    :param generative_model: The generative model to use. Available options are 'mistral', 'falcon', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', and 'command'.
    :param prompt_template: The prompt template to use.
    :param max_length: The maximum length of the generated text.
    """

    model = Model(generative_model=generative_model)
    
    prompt_node = PromptNode(model_name_or_path=model.model_name,
                             api_key=model.api_key,
                             default_prompt_template=prompt_template,
                             max_length=max_length,
                             )

    return prompt_node