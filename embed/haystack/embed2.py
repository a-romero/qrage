import os
from haystack.document_stores import WeaviateDocumentStore, FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http

#faiss_document_store = FAISSDocumentStore(sql_url="sqlite:///", faiss_index_factory_str="Flat")
weaviate_document_store = WeaviateDocumentStore(host='http://localhost',
                                       port=8080,
                                       embedding_dim=1536,
                                       index='got',
                                       recreate_index=True
                                       )

doc_dir = "data/tutorial6"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt6.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)
#doc_dir = "data/misc"


# Convert files to dicts
#docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
docs = convert_files_to_docs(dir_path=doc_dir)


weaviate_document_store.write_documents(docs)

retriever = EmbeddingRetriever(document_store=weaviate_document_store,
                                      embedding_model="text-embedding-ada-002",
                                      api_key=os.getenv('OPENAI_API_KEY'),
                                      top_k=20,
                                      max_seq_len=8191
                                      )
#retriever = EmbeddingRetriever(
#    document_store=weaviate_document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
#)
weaviate_document_store.update_embeddings(retriever)
