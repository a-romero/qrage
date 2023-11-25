from embed import haystack_embed
from query import haystack_query

path_doc = ["data"]
index_name = "revolut"

def main():

    haystack_embed.embed(source=path_doc, index_name=index_name, recreate_index=True)
    haystack_query.query("How would Revolut be impacted by AIG going bankrupt?", index_name=index_name)

if __name__ == '__main__':
    main()