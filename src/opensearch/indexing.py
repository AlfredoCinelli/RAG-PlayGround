"""
Module defining a class to create, delete, and interact with an OpenSearch index.

The main class is OpenSearchIndex with main methods:

* load_index_config: loads the index configuration from a JSON file
* create_index: creates an OpenSearch index based on the configuration
* delete_index: deletes an OpenSearch index
* bulk_index_documents: index multiple documents in OpenSearch index
* delete_documents_by_name: delete documents from OpenSearch index by name
"""

# Import packages and modules

import json
from typing import Any, Dict, List, Tuple

from opensearchpy import helpers
from typing_extensions import Self

from src.constants import (
    ASSYMETRIC_EMBEDDING,
    EMBEDDING_DIMENSION,
    INDEX_CONFIG_PATH,
    OPENSEARCH_INDEX,
)
from src.logging import logger
from src.opensearch import OpenSearchClient

# Define classes


class OpenSearchIndex(OpenSearchClient):
    """Class to create, delete, and interact with an OpenSearch index."""

    def __init__(
        self: Self,
    ) -> None:
        """
        Initialize the OpenSearchIndex class.

        :param self: instance variable
        :type self: Self
        """
        super().__init__()

    @staticmethod
    def load_index_config() -> Dict[str, Any]:
        """
        Method to load the index configuration from a JSON file.

        :return: Dict[str, Any]: the index configuration as a dictionary.
        :rtype: Dict[str, Any]
        """
        with open(INDEX_CONFIG_PATH) as f:
            config = json.load(f)

        # Replace the placeholder with the actual embedding dimension
        config["mappings"]["properties"]["embedding"]["dimension"] = EMBEDDING_DIMENSION
        logger.info(f"Index configuration loaded from {INDEX_CONFIG_PATH}.")
        return config if isinstance(config, dict) else {}

    def create_index(self: Self) -> None:
        """
        Method that creates an index in OpenSearch using settings and mappings from the configuration file.

        :param client: OpenSearch client instance
        :type client: OpenSearch
        :return: None
        :rtype: None
        """
        # Load the index configuration from the JSON file
        index_body = self.load_index_config()
        if not self.client.indices.exists(index=OPENSEARCH_INDEX):
            response = self.client.indices.create(index=OPENSEARCH_INDEX, body=index_body)
            logger.info(f"Created index {OPENSEARCH_INDEX}: {response}")
        else:
            logger.info(f"Index {OPENSEARCH_INDEX} already exists.")

    def delete_index(self: Self) -> None:
        """
        Method to delete the index in OpenSearch if it exists.

        :param client: OpenSearch client instance
        :type client: OpenSearch
        :return: None
        :rtype: None
        """
        if self.client.indices.exists(index=OPENSEARCH_INDEX):
            response = self.client.indices.delete(index=OPENSEARCH_INDEX)
            logger.info(f"Deleted index {OPENSEARCH_INDEX}: {response}")
        else:
            logger.info(f"Index {OPENSEARCH_INDEX} does not exist.")

    def bulk_index_documents(
        self: Self,
        documents: List[Dict[str, Any]],
    ) -> Tuple[int, List[Any]]:
        """
        Method to index multiple documents into OpenSearch in bulk.

        :param documents: list of document dictionaries with 'doc_id', 'text', 'embedding', and 'document_name'
        :type documents: List[Dict[str, Any]]
        :return: tuple with the number of successfully indexed documents and a list of any errors
        :rtype: Tuple[int, List[Any]]
        """
        actions = []

        for doc in documents:
            doc_id = doc["doc_id"]
            embedding_list = doc["embedding"].tolist()
            document_name = doc["document_name"]

            # Prefix each document's text with "passage: " for the asymmetric embedding model
            prefixed_text = f"passage: {doc['text']}" if ASSYMETRIC_EMBEDDING else f"{doc['text']}"

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
        success, errors = helpers.bulk(self.client, actions)
        logger.info(
            f"Bulk indexed {len(documents)} documents into index {OPENSEARCH_INDEX} with {len(errors)} errors."
        )
        return success, errors

    def delete_documents_by_name(
        self: Self,
        document_name: str,
    ) -> Dict[str, Any]:
        """
        Method to delete documents from OpenSearch where 'document_name' matches the provided value.

        :param document_name: name of the document to delete
        :type document_name: str
        :return: response from the delete-by-query operation
        :rtype: Dict[str, Any]
        """
        query = {"query": {"term": {"document_name": document_name}}}
        response: Dict[str, Any] = self.client.delete_by_query(index=OPENSEARCH_INDEX, body=query)
        logger.info(f"Deleted documents with name '{document_name}' from index {OPENSEARCH_INDEX}.")
        return response
