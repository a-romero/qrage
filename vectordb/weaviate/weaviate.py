import os
from typing import List, Union
import weaviate
from weaviate.embedded import EmbeddedOptions

class WeaviateClient:

    def __init__(self,
                url: Union[str, List[str]] = "http://localhost:8080",
                index: str = "Revolut",
                embedding_size: int = 768,
                similarity: str = "cosine",
                index_type: str = "hnsw",
                recreate_index: bool = False,
    ):
        """
        :param url: Weaviate server connection URL for storing and processing documents and vectors.
        :param embedding_size: The embedding vector size. Default: 768.
        :param similarity: The similarity function used to compare document vectors. Available options are 'cosine' (default), 'dot_product', and 'l2'.
        :param index_type: Index type of any vector object defined in the Weaviate schema.
        :param recreate_index: If set to True, deletes an existing Weaviate index and creates a new one using the config you are using for initialization. Note that all data in the old index is
            lost if you choose to recreate the index.
        """
    client = weaviate.Client(
        url=os.getenv('WEAVIATE_ADDRESS'), 
        additional_headers={
            "X-Cohere-Api-Key": os.getenv('COHERE_API_KEY'),
            "X-HuggingFace-Api-Key": os.getenv('HUGGINGFACE_API_KEY'),
            "X-OpenAI-Api-Key": os.getenv('OPENAI_API_KEY'),
        },
        embedded_options=EmbeddedOptions(),
    )

