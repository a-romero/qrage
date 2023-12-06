from utils import files
import weaviate
from haystack.document_stores import WeaviateDocumentStore
from llama_index.vector_stores import WeaviateVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore
from llama_index import SimpleDirectoryReader, VectorStoreIndex, KnowledgeGraphIndex



def kg_index(source: [str],
             space_name: str,
             ):

    documents = SimpleDirectoryReader(source).load_data()
    edge_types, rel_prop_names = ["relationship"], ["relationship"]
    tags = ["entity"]

    graph_store = NebulaGraphStore(space_name=space_name,
                                    edge_types=edge_types,
                                    rel_prop_names=rel_prop_names,
                                    tags=tags,
                                    )
    storage_context = StorageContext.from_defaults()

    kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=10,
        space_name=space_name,
        edge_types=edge_types,
        rel_prop_names=rel_prop_names,
        tags=tags,
        include_embeddings=True,
    )
    kg_index.storage_context.persist(persist_dir='./storage_graph/Revolut')

    print("Knowledge Graph index: ", kg_index)