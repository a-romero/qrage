import os
import logging
import wandb
from components.generators.prompts.prompts import find_prompt_by_id
from components.generators.prompts.haystack_prompt import createPromptNode
from haystack import Pipeline
from haystack.document_stores import WeaviateDocumentStore, InMemoryDocumentStore
from haystack.nodes import AnswerParser, EmbeddingRetriever, PromptTemplate, TopPSampler
from haystack.nodes.ranker import CohereRanker, LostInTheMiddleRanker
from haystack.nodes.retriever.web import WebRetriever
from haystack.utils import print_answers
from wandb.integration.cohere import autolog

def generateWithVectorDB(query: str,
                         prompt_id: str,
                         index_name: str,
                         embedding_model: str="sentence-transformer",
                         dim: int=768,
                         similarity: str="cosine",
                         generative_model: str="gpt-4",
                         top_k: int=5,
                         reranker: str="none",
                         max_length: int=600,
                         debug: bool=False,
                         draw_pipeline: bool=False):
    """
    Processes query from a provided index and a combination of configurable retrieval strategies

    :param query: The input to pass on to the model to use with the prompt template.
    :param prompt_id: The id of the prompt to be used as per the generate/prompt/templates/prompts.csv
    :param model: Embedding model to use. Options are: sentence-transformer, ada.
    :param dim: Number of vector dimensions for the embeddings.
    :param similarity: Similarity function for vector search.
    :param generative_model: Generative model to use. Options are: mistral, gpt-3.5-turbo, gpt-4, gpt-4-turbo, command.
    :param top_k: The top_k parameter defines the number of tokens with the highest probabilities the next token is selected from.
    :param reranker: Whether to use a ReRanker or not. Options are: none, cohere-ranker.
    :param max_length: Length of the response in tokens.
    :param debug: Enable debug on the pipeline.
    :param draw_pipeline: Whether to export a png of the pipeline to the ./diagrams directory.
    """

    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
    logging.getLogger("haystack").setLevel(logging.INFO)
    
    autolog({"project":"qrage", "job_type": "introduction"})

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

    rag_pipeline = Pipeline()

    prompt = find_prompt_by_id(prompt_id)
    if not prompt:
        print("No prompt found for provided id, using default")
        prompt = """"Given the provided Documents, answer the Query.\n
                    Query: {query}\n
                    Documents: {join(documents)}
                    Answer: 
                """    
    print("Using prompt: ", prompt)    
    
    prompt_template = PromptTemplate(prompt = prompt,
                                     output_parser=AnswerParser())
    
    prompt_node=createPromptNode(generative_model, prompt_template ,max_length)
    print("Prompt: ", prompt_node)

    rag_pipeline.add_node(component=embedding_retriever, name="Retriever", inputs=["Query"])

    prompt_input="Retriever"
    if reranker=='cohere-ranker':
        ranker = CohereRanker(model_name_or_path="rerank-english-v2.0",
                              api_key=os.getenv('COHERE_API_KEY'
                              )
        )
        rag_pipeline.add_node(component=ranker, name="Ranker", inputs=[prompt_input])
        print("Ranker: ", ranker)
        prompt_input = "Ranker"
        
    rag_pipeline.add_node(component=prompt_node, name="PromptNode", inputs=[prompt_input])

    response = rag_pipeline.run(query = query, params={"Retriever" : {"top_k": top_k}, "debug": debug})
    
    print("Answer: ", response)

    if draw_pipeline:
        rag_pipeline.draw("diagrams/generative_pipeline.png")
    
    wandb.finish()


def generateWithWebsite(query: str,
                        prompt_id: str,
                        domains: [str]=["google.com"],
                        generative_model: str="gpt-4",
                        top_k: int=5,
                        litm_ranker: bool=True,
                        word_count_threshold: int=1024,
                        max_length: int=600,
                        debug: bool=False,
                        draw_pipeline: bool=False):
    """
    Processes query from a provided website and a combination of configurable retrieval strategies

    :param query: The input to pass on to the model to use with the prompt template.
    :param prompt_id: The id of the prompt to be used as per the generate/prompt/templates/prompts.csv
    :param domains: List of domains to search on.
    :param generative_model: Generative model to use. Options are: mistral, gpt-3.5-turbo, gpt-4, gpt-4-turbo, command.
    :param top_k: Number of top search results to be retrieved.
    :param litm_ranker: Whether to use a Lost In the Middle Ranker or not.
    :param word_count_threshold: The maximum total number of words across all documents selected by the ranger.
    :param max_length: Length of the response in tokens.
    :param debug: Enable debug on the pipeline.
    :param draw_pipeline: Whether to export a png of the pipeline to the ./diagrams directory.
    """

    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
    logging.getLogger("haystack").setLevel(logging.INFO)

    rag_pipeline = Pipeline()

    web_retriever = WebRetriever(
        api_key=os.getenv('SERPERDEV_API_KEY'),
        allowed_domains=domains,
        top_search_results=10,
        mode="preprocessed_documents",
        top_k=top_k,
        cache_document_store=InMemoryDocumentStore(),
    )
    
    prompt = find_prompt_by_id(prompt_id)
    if not prompt:
        print("No prompt found for provided id, using default")
        prompt = """"Given the provided Documents, answer the Query.\n
                    Query: {query}\n
                    Documents: {join(documents)}
                    Answer: 
                """
    print("Using prompt: ", prompt)    
    prompt_template = PromptTemplate(prompt = prompt,
                                     output_parser=AnswerParser())
    
    prompt_node=createPromptNode(generative_model, prompt_template ,max_length)
    print("Prompt: ", prompt_node)

    rag_pipeline.add_node(component=web_retriever, name="Retriever", inputs=["Query"])
    rag_pipeline.add_node(component=TopPSampler(top_p=0.90), name="Sampler", inputs=["Retriever"])

    prompt_input = "Retriever"
    if litm_ranker:
        ranker =LostInTheMiddleRanker(word_count_threshold=word_count_threshold)
        rag_pipeline.add_node(component=ranker, name="LostInTheMiddleRanker", inputs=["Sampler"])
        print("Ranker: ", ranker)
        prompt_input = "LostInTheMiddleRanker"

    rag_pipeline.add_node(component=prompt_node, name="PromptNode", inputs=[prompt_input])
    
    response = rag_pipeline.run(query = query, params={"debug": debug})
    print("Answer: ", response)

    if draw_pipeline:
        rag_pipeline.draw("diagrams/generative_pipeline.png")

