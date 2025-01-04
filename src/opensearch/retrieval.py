"""
Module defining a class to perform different types of search on an OpenSearch index.

The main class is OpenSearchRetriever with main methods:

* lexical_search: performs a lexical search on an OpenSearch index
* vector_search: performs a vector search on an OpenSearch index
* hybrid_search: performs a hybrid search on an OpenSearch index
"""

# Import packages and modules

from typing import Any, Dict, List

from typing_extensions import Self

from src.constants import (
    OPENSEARCH_INDEX,
)
from src.logging import logger
from src.opensearch import OpenSearchClient

# Define Retriever class


class OpenSearchRetriever(OpenSearchClient):
    """Class to perform retrieval using OpenSearch."""

    def __init__(
        self: Self,
    ) -> None:
        """
        Initialize the OpenSearchRetriever class.

        :param self: instance variable
        :type self: Self
        """
        super().__init__()

    def lexical_search(
        self: Self,
        query_text: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Method that performs a lexical search using BM25.

        :param self: instance variable
        :type self: Self
        :param query_text: the user text query for text-based search
        :type query_text: str
        :param top_k: the number of search results to return, defaults to 5
        :type top_k: int, optional
        :return: a list of serach results from OpenSearch
        :rtype: List[Dict[str, Any]]
        """
        query_body = {
            "_source": {"exclude": ["embedding"]},  # Exclude embeddings from the results
            "query": {
                "match": {  # Text-based search (BM25)
                    "text": {
                        "query": query_text,
                    }
                }
            },
            "size": top_k,
        }

        response = self.client.search(
            index=OPENSEARCH_INDEX,  # specify the index to search in
            body=query_body,  # specify the query body
            search_pipeline="nlp-search-pipeline",  # specify the search pipeline
        )
        logger.info(
            f"Lexical search completed for query '{query_text}' with {top_k} retrieved results."
        )

        # Type casting for compatibility with expected return type
        hits: List[Dict[str, Any]] = response["hits"]["hits"]
        return hits

    def vector_search(
        self: Self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Method that performs a vector search using ANN.
        To have more info about the ANN algorithm refer to the OpenSearch
        configuration file (config/index_config.json).

        :param self: instance variable
        :type self: Self
        :param query_text: the user text query for text-based search
        :type query_text: str
        :param query_embedding: the embedding vector for vector-based search
        :type query_embedding: List[float]
        :param top_k: the number of top results to retrieve, defaults to 5
        :type top_k: int, optional
        :return: a list of serach results from OpenSearch
        :rtype: List[Dict[str, Any]]
        """
        query_body = {
            "_source": {"exclude": ["embedding"]},  # Exclude embeddings from the results
            "query": {
                "knn": {  # Vector search (approximate KNN)
                    "embedding": {
                        "vector": query_embedding,
                        "k": top_k,
                    }
                }
            },
            "size": top_k,
        }

        response = self.client.search(
            index=OPENSEARCH_INDEX,  # specify the index to search in
            body=query_body,  # specify the query body
            search_pipeline="nlp-search-pipeline",  # specify the search pipeline
        )
        logger.info(
            f"Vector search completed for query '{query_text}' with {top_k} retrieved results."
        )

        # Type casting for compatibility with expected return type
        hits: List[Dict[str, Any]] = response["hits"]["hits"]
        return hits

    def hybrid_search(
        self: Self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Method that performs a hybrid search combining text-based and vector-based queries
        with respective BM25 and ANN algorithms.

        :param self: instance variable
        :type self: Self
        :param query_text: the user text query for text-based search
        :type query_text: str
        :param query_embedding: the embedding vector for vector-based search
        :type query_embedding: List[float]
        :param top_k: the number of top results to retrieve, defaults to 5
        :type top_k: int, optional
        :return: a list of serach results from OpenSearch
        :rtype: List[Dict[str, Any]]
        """
        query_body = {
            "_source": {"exclude": ["embedding"]},  # Exclude embeddings from the results
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "match": {  # Text-based search (BM25)
                                "text": {
                                    "query": query_text,
                                }
                            }
                        },
                        {
                            "knn": {  # Vector search (approximate KNN)
                                "embedding": {
                                    "vector": query_embedding,
                                    "k": top_k,
                                }
                            }
                        },
                    ]
                }
            },
            "size": top_k,
        }

        response = self.client.search(
            index=OPENSEARCH_INDEX,  # specify the index to search in
            body=query_body,  # specify the query body
            search_pipeline="nlp-search-pipeline",  # specify the search pipeline
        )
        logger.info(
            f"Hybrid search completed for query '{query_text}' with {top_k} retrieved results."
        )

        # Type casting for compatibility with expected return type
        hits: List[Dict[str, Any]] = response["hits"]["hits"]
        return hits
