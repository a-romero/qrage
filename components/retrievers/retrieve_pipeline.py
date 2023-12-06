import string
from components.retrievers import custom_retriever
from components.generators.models import Model

from llama_index import (
    ServiceContext,
    VectorStoreIndex
)
import weaviate

from llama_index import load_index_from_storage
from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore
from llama_index.vector_stores import WeaviateVectorStore

# Retrievers
from llama_index.retrievers import BaseRetriever, VectorIndexRetriever, KGTableRetriever
from llama_index import get_response_synthesizer
from llama_index.query_engine import RetrieverQueryEngine


def get_response_with_VKBRetriever(
        user_question: str,
        index_name: str,
        space_name: str,
        generative_model: str,
        temperature: int=0
):
    """
    :param user_question: user question
    :param index_name: index name
    :param space_name: space name
    :param generative_model: model name
    :param temperature: temperature
    """

    graph_storage_dir = "./storage_graph/"
    persist_dir = graph_storage_dir + space_name
    
    edge_types, rel_prop_names = ["relationship"], ["relationship"]
    tags = ["entity"]

    weaviate_client = weaviate.Client("http://localhost:8080")
    vector_store = WeaviateVectorStore(weaviate_client=weaviate_client, index_name=index_name)
    #storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_index = VectorStoreIndex.from_vector_store(vector_store)
    vector_retriever = VectorIndexRetriever(index=vector_index)

    graph_store = NebulaGraphStore(
        space_name=space_name,
        edge_types=edge_types,
        rel_prop_names=rel_prop_names,
        tags=tags,
    )
    graph_storage_context = StorageContext.from_defaults(persist_dir=persist_dir, graph_store=graph_store)

    model = Model(generative_model=generative_model, temperature=temperature)
    llm = model.baseModel(model.model_name)
    
    service_context = ServiceContext.from_defaults(llm=llm, chunk_size_limit=512)
    kg_index = load_index_from_storage(storage_context=graph_storage_context,
                                       service_context=service_context,
                                       max_triplets_per_chunk=10,
                                       space_name=space_name,
                                       edge_types=edge_types,
                                       rel_prop_names=rel_prop_names,
                                       tags=tags,
                                       include_embeddings=True)

    kg_retriever = KGTableRetriever(index=kg_index, retriever_mode="keyword", include_text=False)

    retriever = custom_retriever.VKGRetriever(vector_retriever, kg_retriever)
   

    # create response synthesizer
    response_synthesizer = get_response_synthesizer(
        service_context=service_context,
        response_mode="tree_summarize",
    )

    custom_query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    response = custom_query_engine.query(user_question)

    print(response)