from srcs.rag import qent_retriever

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from llama_index.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from llama_index import (
    KnowledgeGraphIndex,
    LLMPredictor,
    ServiceContext,
    SimpleDirectoryReader,
    VectorStoreIndex
)

from llama_index import load_index_from_storage
from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore

# Retrievers
from llama_index.retrievers import BaseRetriever, VectorIndexRetriever, KGTableRetriever
from llama_index import get_response_synthesizer
from llama_index.query_engine import RetrieverQueryEngine


def get_response(user_question, project):
    edge_types, rel_prop_names = ["relationship"], [
        "relationship"
    ]
    tags = ["entity"]
    
    graph_store = NebulaGraphStore(
        space_name=project,
        edge_types=edge_types,
        rel_prop_names=rel_prop_names,
        tags=tags,
    )
    graph_storage_context = StorageContext.from_defaults(persist_dir='./storage_graph', graph_store=graph_store)
    
    vector_storage_context = StorageContext.from_defaults(persist_dir='./storage_vector')
    
    llm = OpenAI(temperature=0, model="gpt-4")
    
    service_context = ServiceContext.from_defaults(llm=llm, chunk_size_limit=512)

    kg_index = load_index_from_storage(
        storage_context=graph_storage_context,
        service_context=service_context,
        max_triplets_per_chunk=10,
        space_name=project,
        edge_types=edge_types,
        rel_prop_names=rel_prop_names,
        tags=tags,
        verbose=True,
    )

    vector_index = load_index_from_storage(
        service_context=service_context,
        storage_context=vector_storage_context
    )

    kg_retriever = KGTableRetriever(
        index=kg_index, retriever_mode="keyword", include_text=False
    )
    vector_retriever = VectorIndexRetriever(index=vector_index)

    custom_retriever = qent_retriever.qEntRetriever(vector_retriever, kg_retriever)

    # create response synthesizer
    response_synthesizer = get_response_synthesizer(
        service_context=service_context,
        response_mode="tree_summarize",
    )

    custom_query_engine = RetrieverQueryEngine(
        retriever=custom_retriever,
        response_synthesizer=response_synthesizer,
    )

    response = custom_query_engine.query(user_question)

    print(response)