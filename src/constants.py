"""Module containing literals used throughout the application."""

# Embeddings model settings
EMBEDDING_MODEL_PATH = "thenlper/gte-base"  # (gte-large) OR Path of local eg. "embedding_model/"" or the name of SentenceTransformer model eg. "sentence-transformers/all-mpnet-base-v2" from Hugging Face
RERANKER_MODEL_PATH = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # path to the HuggingFace model for reranking (eg. "cross-encoder/ms-marco-MiniLM-L-6-v2")
ASSYMETRIC_EMBEDDING = False  # Flag for asymmetric embedding
EMBEDDING_DIMENSION = (
    768  # Embedding model settings (728 for mpnet-v2 - 768 for gte-base - 1024 for gte-large)
)
TEXT_CHUNK_SIZE = 450  # Maximum number of characters in each text chunk for (align with embedding dimension, for istance 512 token for gte-large)
CHUNK_OVERLAP_SIZE = 50  # Number of characters to overlap between chunks

# LLM settings

# Availabe models in Ollama (pull if not available)
MODEL_NAMES = {
    "GEMMA2 9B": "gemma2",  # general LLM by Google
    "GEMMA2 2B": "gemma2:2b",  # general LLM by Google
    "LLAMA 3.2 3B": "llama3.2",  # general LLM by Meta
    "LLAMA 3.2 1B": "llama3.2:1b",  # general LLM by Meta
    "LLAMA 3.3": "llama3.3",  # general LLM by Meta
    "MISTRAL 7B": "mistral",  # general LLM by Mistral
    "PHI3 3.8B": "phi3:3.8b",  # general LLM by Microsoft
    "LLAVA 7B": "llava",  # general LLM
    "QWEN 1.5B": "qwen2.5-coder:1.5b",  # coding LLM by Alibaba
    "QWEN 7B": "qwen2.5-coder:7b",  # coding LLM by Alibaba
    "QWEN 3B": "qwen2.5-coder:3b",  # coding LLM by Alibaba
    "CODE-GEMMA 2B": "codegemma:2b",  # coding LLM by Google
}

# WebApp
LOGO_PATH = "images/ai-logo.png"

####################################################################################################
# Dont change the following settings
####################################################################################################

# Logging
LOG_FILE_DIR = "logs"  # Name of the logs folder
LOG_FILE_NAME = "app.log"  # Name of the log file
# OpenSearch settings
INDEX_CONFIG_PATH = "conf/index_config.json"  # Path to the index configuration file
OPENSEARCH_HOST = "localhost"  # Hostname for the OpenSearch instance
OPENSEARCH_PORT = 9200  # Port number for OpenSearch
OPENSEARCH_INDEX = "documents"  # Index name for storing documents in OpenSearch
