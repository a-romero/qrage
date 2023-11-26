# qRage - Modular framework for building Retrieval Augmented Generation (RAG) pipelines

## Introduction
This framework provides flexible Retrieval Augmented Generation (RAG) by integrating with frameworks like Haystack and LlamaIndex. It uses parametrized configurations to leverage various transformer and LLM models. Key features include:

- Integration with Weaviate vector database and Nebula graph database.
- Custom retrievers combining Hybrid Search, Embedding Fine-tuning, Generative Pseudo Labelling, and ReRanking techniques.

## Installation
Clone the repository:
```bash
git clone https://github.com/a-romero/qrage.git
cd qrage
```

## 🐳 Starting Weaviate Vector Database
### Standard Setup
1. Navigate to the standard directory:
   ```bash
   cd vectordb/weaviate/standard
   ```
2. Start the Weaviate instance using Docker:
   ```bash
   docker-compose up -d
   ```

### Advanced Setup
1. Navigate to the advanced directory:
   ```bash
   cd vectordb/weaviate/advanced
   ```
2. Start the Weaviate instance using Docker:
   ```bash
   docker-compose up -d
   ```
*Note:* The Weaviate service becomes available locally on http://localhost:8080
## 🐳 Starting Nebula Graph Database
1. Navigate to the standard directory:
   ```bash
   cd graphdb/nebulagraph
   ```
2. Start the Nebula instance using Docker:
   ```bash
   docker-compose up -d
   ```
*Note:* The Nebulagraph service becomes available locally on http://localhost:9669 with the default credentials root/nebula

## Requirements
- OpenAI API key exported to local env variable `OPENAI_API_KEY`
- HuggingFace API key exported to local env variable `HUGGINGFACEHUB_API_TOKEN`
- Cohere API key exported to local env variable `COHERE_API_KEY`

## Usage
### Data sources
The following file formats are supported:
- .text
- .pdf
- .md
- .docx

For data input, the following sources are supported:
- Local file
- Local directory (recursive ingestion)
- HTTP(S)
- S3

### Indexing/Embedding pipeline
The embedding models supported are:
- "sentence-transformer" => "sentence-transformers/multi-qa-mpnet-base-dot-v1"
- (OpenAI) "ada" => "text-embedding-ada-002"
- (Cohere) "embed" => "embed-multilingual-v2.0"

```python
haystack_embed.embed(source="./data", 
                    index_name="test", 
                    recreate_index=True,
                    batch_size=5,
                    model="sentence-transformer",
                    dim=768,
                    language="en"
                    )
```

### Retrieval/Querying pipeline
The generative models supported are:
- (Mistral) "mistral" => "mistralai/Mistral-7B-Instruct-v0.1"
- (OpenAI) "gpt-3.5-turbo"
- (OpenAI) "gpt-4"
- (OpenAI) "gpt-4-turbo" => "gpt-4-1106-preview"
- (Cohere) "command"

```python
haystack_query.query(query="How would Revolut be impacted by AIG going bankrupt?", 
                    index_name="test",
                    embedding_model="sentence-transformer",
                    dim=768,
                    generative_model="gpt-4",
                    top_k=5,
                    draw_pipeline=False
                    )
```