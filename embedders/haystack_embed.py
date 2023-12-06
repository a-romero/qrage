import os
import logging
import torch
from utils import files
from haystack import Pipeline
from haystack.document_stores import WeaviateDocumentStore
from haystack.nodes.question_generator import QuestionGenerator
from haystack.nodes.label_generator import PseudoLabelGenerator
from haystack.nodes import EmbeddingRetriever, MarkdownConverter, PreProcessor, FileTypeClassifier, TextConverter, PDFToTextConverter, DocxToTextConverter

def embed(source: [str],
          index_name: str,
          recreate_index: bool=False,
          batch_size: int=5,
          model: str="sentence-transformer",
          dim: int=768,
          top_k: int=10,
          gpl: bool=False,
          language: str="en"):
    """
    Processes embeddings from a provided array of file/directory paths

    :param source: Array of file/directory paths to embed.
    :param index_name: The name of the index to use in the Weaviate Document Store.
    :param recreate_index: Flag to recreate index.
    :param batch_size: Parallelism of document processing.
    :param model: Embedding model to use. Options are: sentence-transformer, ada, embed.
    :param dim: Number of vector dimensions for the embeddings.
    :param top_k: The top_k parameter defines the number of tokens with the highest probabilities the next token is selected from.
    :param gpl: Enable Generative Pseudo Labeling for domain adaptation through synthetic Q/A.
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

        preprocessor = PreProcessor(clean_empty_lines=True,
                                clean_whitespace=False,
                                clean_header_footer=False,
                                split_by="word",
                                split_length=500,
                                split_respect_sentence_boundary=True,
                                language=language,
        )

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
        
        if gpl:
            if torch.cuda.is_available():
                questions_producer = QuestionGenerator(
                    model_name_or_path="doc2query/msmarco-t5-base-v1",
                    max_length=64,
                    split_length=128,
                    batch_size=32,
                    num_queries_per_doc=3,
                )
                plg = PseudoLabelGenerator(question_producer=questions_producer, retriever=embedding_retriever, max_questions_per_document=10, top_k=top_k)
                output, pipe_id = plg.run(documents=document_store.get_all_documents())
                output["gpl_labels"][0]
                embedding_retriever.train(output["gpl_labels"])
            else:
                print("Skipping Generative Pseudo Labeling as no GPU detected")

        print("Finished processing files\n")

        if draw_pipeline:
            indexing_pipeline.draw("diagrams/indexing_pipeline.png")


    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
    logging.getLogger("haystack").setLevel(logging.INFO)

    document_store = WeaviateDocumentStore(host='http://localhost',
                                          port=8080,
                                          embedding_dim=dim,
                                          index=index_name,
                                          batch_size=batch_size,
                                          recreate_index=recreate_index)

    # Choice of retrievers
    embedding_retriever = EmbeddingRetriever(embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    if model=='ada':
        embedding_retriever = EmbeddingRetriever(document_store=document_store,
                                        embedding_model="text-embedding-ada-002",
                                        api_key=os.getenv('OPENAI_API_KEY'),
                                        top_k=top_k,
                                        max_seq_len=8191
        )
    elif model=='embed':
        embedding_retriever = EmbeddingRetriever(document_store=document_store,
                                        embedding_model="embed-multilingual-v2.0",
                                        api_key=os.getenv('COHERE_API_KEY'),
                                        top_k=top_k
        )
    else:
        embedding_retriever = EmbeddingRetriever(document_store = document_store,
                                        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
                                        model_format="sentence_transformers",
                                        top_k=top_k
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