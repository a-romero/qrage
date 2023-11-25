import os
import sys
from dotenv import load_dotenv

from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore
from llama_index import (
    KnowledgeGraphIndex,
    LLMPredictor,
    ServiceContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
    Document
)

def nebula_init ():
    os.environ["NEBULA_USER"] = "root"
    os.environ["NEBULA_PASSWORD"] = "nebula"
    os.environ[
        "NEBULA_ADDRESS"
    ] = "192.168.1.98:9669"

def nebula_index(project, documents):
    
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
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=10,
        space_name=project,
        edge_types=edge_types,
        rel_prop_names=rel_prop_names,
        tags=tags,
        include_embeddings=True,
    )

    return kg_index