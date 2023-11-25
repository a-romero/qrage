import os
from haystack import Pipeline
from haystack.pipelines import ExtractiveQAPipeline
from haystack.document_stores import WeaviateDocumentStore
from haystack.nodes import AnswerParser, EmbeddingRetriever, PromptNode, PromptTemplate, FARMReader
from haystack.utils import print_answers


def query(query: str,
          index_name: str,
          embedding_model: str="sentence-transformer",
          dim: int=768,
          prompt_model: str="gpt-4",
          top_k: int=5,
          draw_pipeline: bool=False):

    document_store = WeaviateDocumentStore(host='http://localhost',
                                        port=8080,
                                        embedding_dim=dim,
                                        index=index_name)

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
    if prompt_model=='mistral':
        prompt_node = PromptNode(model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.1",
                                 api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN'),
                                 default_prompt_template = prompt_template,
                                 max_length=800
                                 )
    else:
        prompt_node = PromptNode(model_name_or_path = "gpt-4",
                                api_key = os.getenv('OPENAI_API_KEY'),
                                default_prompt_template = prompt_template)

    query = query

    # Simple pipeline passing full docs
    query_pipeline = Pipeline()
    query_pipeline.add_node(component=embedding_retriever, name="Retriever", inputs=["Query"])
    query_pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])

    response = query_pipeline.run(query = query, params={"Retriever" : {"top_k": top_k}})
    print("Answer: ", response)

    if draw_pipeline:
        query_pipeline.draw("diagrams/query_pipeline.png")

