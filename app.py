from components.embedders import haystack_embed, llamaindex_embed
from components.generators import haystack_generate
from components.retrievers import retrieve_pipeline

path_doc = ["data"]
path = "data"
index_name = "Revolut"
domains = ["crunchbase.com"]

def main():

    haystack_embed.embed(source=path_doc,
                         index_name=index_name,
                         model="ada",
                         #dim=1536,
                         preprocess=True,
                         recreate_index=True
    )

    haystack_generate.generateWithVectorDB("How would Revolut be impacted by AIG going bankrupt?",
                                           prompt_id="Business Analyst",
                                           index_name=index_name,
                                           embedding_model="ada",
                                           generative_model="gpt-4-turbo",
                                           reranker="cohere-ranker",
                                           max_length=300
    )
    haystack_generate.generateWithWebsite("Write a brief introduction of Revolut's CEO",
                                          prompt_id="Business Analyst",
                                          domains=domains,
                                          generative_model="gpt-4",
                                          litm_ranker=True,
                                          max_length=800
    )
    
    llamaindex_embed.kg_index(source=path,
                              space_name=index_name)

    retrieve_pipeline.get_response_with_VKBRetriever("How would Revolut be impacted by AIG going bankrupt?",
                                                     prompt_id="Business Analyst",
                                                     generative_model="gpt-4",
                                                     index_name=index_name,
                                                     space_name=index_name
                                                     )

if __name__ == '__main__':
    main()