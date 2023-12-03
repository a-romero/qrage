from embed import haystack_embed
from generate import haystack_generate

path_doc = ["data"]
index_name = "revolut"
domains = ["crunchbase.com"]

def main():

    haystack_embed.embed(source=path_doc,
                         index_name=index_name,
                         recreate_index=True
    )
    haystack_generate.generateWithVectorDB("How would Revolut be impacted by AIG going bankrupt?",
                                           prompt_id="Business Analyst",
                                           index_name=index_name,
                                           generative_model="gpt-4",
                                           reranker="cohere-ranker",
                                           max_length=800
    )
    haystack_generate.generateWithWebsite("Write a brief introduction of Revolut's CEO",
                                          prompt_id="Business Analyst",
                                          domains=domains,
                                          generative_model="gpt-4",
                                          litm_ranker=True,
                                          max_length=800
    )

if __name__ == '__main__':
    main()