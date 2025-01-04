"""
Module containig functions to perform ingestion of data into OpenSearch.

The main functions defined in the module are:

* load_index_config: Loads the index configuration from a JSON file.
* create_index: Creates an OpenSearch index with the specified configuration.
* delete_index: Deletes the specified index from OpenSearch.
* bulk_index_documents: Indexes a list of documents into the specified index.
"""

import json
from typing import Any, Dict, List, Tuple

from opensearchpy import OpenSearch, helpers

from src.constants import (
    ASSYMETRIC_EMBEDDING,
    EMBEDDING_DIMENSION,
    INDEX_CONFIG_PATH,
    OPENSEARCH_INDEX,
)
from src.logging import logger
from src.opensearch import get_opensearch_client


def load_index_config() -> Dict[str, Any]:
    """
    Function to load the index configuration from a JSON file.

    :return: Dict[str, Any]: the index configuration as a dictionary.
    :rtype: Dict[str, Any]
    """
    with open(INDEX_CONFIG_PATH) as f:
        config = json.load(f)

    # Replace the placeholder with the actual embedding dimension
    config["mappings"]["properties"]["embedding"]["dimension"] = EMBEDDING_DIMENSION
    logger.info(f"Index configuration loaded from {INDEX_CONFIG_PATH}.")
    return config if isinstance(config, dict) else {}


def create_index(client: OpenSearch) -> None:
    """
    Function that creates an index in OpenSearch using settings and mappings from the configuration file.

    :param client: OpenSearch client instance
    :type client: OpenSearch
    :return: None
    :rtype: None
    """
    index_body = load_index_config()
    if not client.indices.exists(index=OPENSEARCH_INDEX):
        response = client.indices.create(index=OPENSEARCH_INDEX, body=index_body)
        logger.info(f"Created index {OPENSEARCH_INDEX}: {response}")
    else:
        logger.info(f"Index {OPENSEARCH_INDEX} already exists.")


def delete_index(client: OpenSearch) -> None:
    """
    Function to delete the index in OpenSearch if it exists.

    :param client: OpenSearch client instance
    :type client: OpenSearch
    :return: None
    :rtype: None
    """
    if client.indices.exists(index=OPENSEARCH_INDEX):
        response = client.indices.delete(index=OPENSEARCH_INDEX)
        logger.info(f"Deleted index {OPENSEARCH_INDEX}: {response}")
    else:
        logger.info(f"Index {OPENSEARCH_INDEX} does not exist.")


def bulk_index_documents(documents: List[Dict[str, Any]]) -> Tuple[int, List[Any]]:
    """
    Function to index multiple documents into OpenSearch in bulk.

    :param documents: list of document dictionaries with 'doc_id', 'text', 'embedding', and 'document_name'
    :type documents: List[Dict[str, Any]]
    :return: tuple with the number of successfully indexed documents and a list of any errors
    :rtype: Tuple[int, List[Any]]
    """
    actions = []
    client = get_opensearch_client()

    for doc in documents:
        doc_id = doc["doc_id"]
        embedding_list = doc["embedding"].tolist()
        document_name = doc["document_name"]

        # Prefix each document's text with "passage: " for the asymmetric embedding model
        if ASSYMETRIC_EMBEDDING:
            prefixed_text = f"passage: {doc['text']}"
        else:
            prefixed_text = f"{doc['text']}"

        action = {
            "_index": OPENSEARCH_INDEX,
            "_id": doc_id,  # Document unique identifier
            "_source": {
                "text": prefixed_text,  # Document chunks
                "embedding": embedding_list,  # Precomputed embedding
                "document_name": document_name,
            },
        }
        actions.append(action)

    # Perform bulk indexing and capture response details explicitly
    success, errors = helpers.bulk(client, actions)
    logger.info(
        f"Bulk indexed {len(documents)} documents into index {OPENSEARCH_INDEX} with {len(errors)} errors."
    )
    return success, errors


def delete_documents_by_document_name(document_name: str) -> Dict[str, Any]:
    """
    Function to delete documents from OpenSearch where 'document_name' matches the provided value.

    :param document_name: name of the document to delete
    :type document_name: str
    :return: response from the delete-by-query operation
    :rtype: Dict[str, Any]
    """
    client = get_opensearch_client()
    query = {"query": {"term": {"document_name": document_name}}}
    response: Dict[str, Any] = client.delete_by_query(
        index=OPENSEARCH_INDEX, body=query
    )
    logger.info(
        f"Deleted documents with name '{document_name}' from index {OPENSEARCH_INDEX}."
    )
    return response
