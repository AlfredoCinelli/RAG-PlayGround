"""
This package contains the core functionality of the project.

More in details:

* chat.py: Module containing functions to create prompt and generate responses from Ollama.
* constants.py: Module containing constants used throughout the project.
* embeddings.py: Module containing functions to generate embeddings.
* opensearch.py: Module containing functions to interact with OpenSearch.
* ocr.py: Module containing functions to perform documents loading and OCR.
* ingestion.py: Module containing functions to perform ingestion of data into OpenSearch.
* utils.py: Module containing utility functions.
"""

from dotenv import load_dotenv

load_dotenv("conf/.env")