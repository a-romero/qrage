import os
from haystack import Pipeline
from haystack.document_stores import WeaviateDocumentStore
from haystack.nodes import AnswerParser, EmbeddingRetriever, PromptNode, PromptTemplate, CohereRanker
from haystack.utils import print_answers


def query(query: str,
          index_name: str,
          embedding_model: str="sentence-transformer",
          dim: int=768,
          similarity: str="cosine",
          generative_model: str="gpt-4",
          top_k: int=5,
          reranker: str="none",
          max_length: int=600,
          draw_pipeline: bool=False):
    """
    Processes query from a provided index and a combination of configurable retrieval strategies

    :param index_name: The name of the index to use in the Weaviate Document Store.
    :param model: Embedding model to use. Options are: sentence-transformer, ada.
    :param dim: Number of vector dimensions for the embeddings.
    :param similarity: Similarity function for vector search.
    :param generative_model: Generative model to use. Options are: mistral, gpt-3.5-turbo, gpt-4, gpt-4-turbo, command.
    :param top_k: The top_k parameter defines the number of tokens with the highest probabilities the next token is selected from.
    :param reranker: Whether to use a ReRanker or not. Options are: none, cohere-ranker.
    :param max_length: Length of the response in tokens.
    :param draw_pipeline: Whether to export a png of the pipeline to the ./diagrams directory.
    """

    document_store = WeaviateDocumentStore(host='http://localhost',
                                        port=8080,
                                        embedding_dim=dim,
                                        index=index_name,
                                        similarity=similarity)

     # Choice of retrievers
    embedding_retriever = EmbeddingRetriever(embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    if embedding_model=='ada':
        embedding_retriever = EmbeddingRetriever(document_store=document_store,
                                        embedding_model="text-embedding-ada-002",
                                        api_key=os.getenv('OPENAI_API_KEY'),
                                        top_k=20,
                                        max_seq_len=8191
                                        )
    else:
        embedding_retriever = EmbeddingRetriever(document_store = document_store,
                                        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
                                        model_format="sentence_transformers",
                                        top_k=20
                                        )   

    print("Retriever: ", embedding_retriever)


    prompt_template = PromptTemplate(prompt = """"Given the provided Documents, answer the Query.\n
                                                Query: {query}\n
                                                Documents: {join(documents)}
                                                Answer: 
                                            """,
                                            output_parser=AnswerParser())
    # Choice of prompt models
    prompt_node = PromptNode()
    if generative_model=='mistral':
        prompt_node = PromptNode(model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.1",
                                 api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN'),
                                 default_prompt_template = prompt_template,
                                 max_length=max_length
        )
    elif generative_model=='gpt-3.5-turbo':
        prompt_node = PromptNode(model_name_or_path = "gpt-3.5-turbo",
                                api_key = os.getenv('OPENAI_API_KEY'),
                                default_prompt_template = prompt_template,
                                max_length=max_length
        )
    elif generative_model=='gpt-4-turbo':
        prompt_node = PromptNode(model_name_or_path = "gpt-4-1106-preview",
                                api_key = os.getenv('OPENAI_API_KEY'),
                                default_prompt_template = prompt_template,
                                max_length=max_length
        )
    elif generative_model=='command':
        prompt_node = PromptNode(model_name_or_path = "command",
                                api_key = os.getenv('COHERE_API_KEY'),
                                default_prompt_template = prompt_template,
                                max_length=max_length
        )
    else:
        prompt_node = PromptNode(model_name_or_path = "gpt-4",
                                api_key = os.getenv('OPENAI_API_KEY'),
                                default_prompt_template = prompt_template,
                                max_length=max_length
        )


    print("Prompt: ", prompt_node)

    # Simple pipeline passing full docs
    query_pipeline = Pipeline()
    query_pipeline.add_node(component=embedding_retriever, name="Retriever", inputs=["Query"])

    prompt_input="Retriever"
    if reranker=='cohere-ranker':
        ranker = CohereRanker(model_name_or_path="rerank-english-v2.0",
                              api_key=os.getenv('COHERE_API_KEY'
                              )
        )
        query_pipeline.add_node(component=ranker, name="Ranker", inputs=[prompt_input])
        print("Ranker: ", ranker)
        prompt_input = "Ranker"
        
    query_pipeline.add_node(component=prompt_node, name="PromptNode", inputs=[prompt_input])

    response = query_pipeline.run(query = query, params={"Retriever" : {"top_k": top_k}})
    print("Answer: ", response)

    if draw_pipeline:
        query_pipeline.draw("diagrams/query_pipeline.png")

