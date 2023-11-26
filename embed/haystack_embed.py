import os
from utils import files
from haystack import Pipeline
from haystack.document_stores import WeaviateDocumentStore
from haystack.nodes import EmbeddingRetriever, MarkdownConverter, PreProcessor, FileTypeClassifier, TextConverter, PDFToTextConverter, DocxToTextConverter


def embed(source: [str],
          index_name: str,
          recreate_index: bool=False,
          batch_size: int=5,
          model: str="sentence-transformer",
          dim: int=768,
          language: str="en"):
    """
    Processes embeddings from a provided array of file/directory paths

    :param source: Array of file/directory paths to embed.
    :param index_name: The name of the index to use in the Weaviate Document Store.
    :param recreate_index: Flag to recreate index.
    :param batch_size: Parallelism of document processing.
    :param model: Embedding model to use. Options are: sentence-transformer, ada, embed.
    :param dim: Number of vector dimensions for the embeddings.
    :param language: Language of the text.
    """

    def embed_paths(file_paths, draw_pipeline=False):

        file_type_classifier = FileTypeClassifier()

        text_converter = TextConverter()
        pdf_converter = PDFToTextConverter()
        md_converter = MarkdownConverter()
        docx_converter = DocxToTextConverter()
        indexing_pipeline = Pipeline()

        indexing_pipeline.add_node(component=file_type_classifier, name="FileTypeClassifier", inputs=["File"])
        indexing_pipeline.add_node(component=text_converter, name="TextConverter", inputs=["FileTypeClassifier.output_1"])
        indexing_pipeline.add_node(component=pdf_converter, name="PDFToTextConverter", inputs=["FileTypeClassifier.output_2"])
        indexing_pipeline.add_node(component=md_converter, name="MarkdownConverter", inputs=["FileTypeClassifier.output_3"])
        indexing_pipeline.add_node(component=docx_converter, name="DocxConverter", inputs=["FileTypeClassifier.output_4"])

        indexing_pipeline.add_node(component=preprocessor, name="PreProcessor", inputs=["TextConverter", 
                                                                                        "PDFToTextConverter", 
                                                                                        "MarkdownConverter", 
                                                                                        "DocxConverter"],
                                                                                        )

        indexing_pipeline.add_node(component=embedding_retriever, name="EmbeddingRetriever", inputs=["PreProcessor"])
        indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["EmbeddingRetriever"])
        
        print("\nProcessing files:")
        for file in file_paths:
            print(file)

        indexing_pipeline.run(file_paths=file_paths)
        print("Finished processing files\n")

        if draw_pipeline:
            indexing_pipeline.draw("diagrams/indexing_pipeline.png")


    document_store = WeaviateDocumentStore(host='http://localhost',
                                          port=8080,
                                          embedding_dim=dim,
                                          index=index_name,
                                          batch_size=batch_size,
                                          recreate_index=recreate_index)

    preprocessor = PreProcessor(clean_empty_lines=True,
                                clean_whitespace=False,
                                clean_header_footer=False,
                                split_by="word",
                                split_length=500,
                                split_respect_sentence_boundary=True,
                                language=language,
    )

    # Choice of retrievers
    embedding_retriever = EmbeddingRetriever(embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    if model=='ada':
        embedding_retriever = EmbeddingRetriever(document_store=document_store,
                                        embedding_model="text-embedding-ada-002",
                                        api_key=os.getenv('OPENAI_API_KEY'),
                                        top_k=20,
                                        max_seq_len=8191
        )
    elif model=='embed':
        embedding_retriever = EmbeddingRetriever(document_store=document_store,
                                        embedding_model="embed-multilingual-v2.0",
                                        api_key=os.getenv('COHERE_API_KEY'),
                                        top_k=20
        )
    else:
        embedding_retriever = EmbeddingRetriever(document_store = document_store,
                                        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
                                        model_format="sentence_transformers",
                                        top_k=20
        )   

    print("Retriever: ", embedding_retriever)

    processed_paths = files.process_paths(source)
    txt_files, pdf_files, md_files, docx_files = files.classify_files(processed_paths)
    
    if txt_files:
        embed_paths(txt_files)
    
    if pdf_files:
        embed_paths(pdf_files)

    if md_files:
        embed_paths(md_files)

    if docx_files:
        embed_paths(docx_files)

    print("#####################\nEmbeddings Completed.")

