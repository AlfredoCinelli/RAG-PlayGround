"""
Module containig functions to interact with OpenSearch.

The main functions of the module are:

* get_opensearch_client: instantiate and return an OpenSearch client.
* lexical_search: perform a lexical search on OpenSearch.
* vector_search: perform a vector search on OpenSearch.
* hybrid_search: perform a hybrid search on OpenSearch.
"""

# Import packages and modules

from typing import Any, Dict, List

from opensearchpy import OpenSearch

from src.constants import OPENSEARCH_HOST, OPENSEARCH_INDEX, OPENSEARCH_PORT
from src.logging import logger

#  Define UDFs


def get_opensearch_client() -> OpenSearch:
    """
    Function to initialize and returns an OpenSearch client.
    The function ensable a lexical search via Okapi BM25 and
    semantic search via ANN.

    :return: OpenSearch: configured OpenSearch client instance
    :rtype: OpenSearch
    """
    client = OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_compress=True,
        timeout=30,
        max_retries=3,
        retry_on_timeout=True,
    )
    logger.info("OpenSearch client initialized.")
    return client


def lexical_search(
    query_text: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Function that performs a lexical search using BM25.

    :param query_text: the user text query for text-based search
    :type query_text: str
    :param top_k: the number of search results to return, defaults to 5
    :type top_k: int, optional
    :return: a list of serach results from OpenSearch
    :rtype: List[Dict[str, Any]]
    """
    client = get_opensearch_client()

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

    response = client.search(
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
    query_text: str,
    query_embedding: List[float],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Function that performs a vector search using L2 distance and ANN.

    :param query_text: the user text query for text-based search
    :type query_text: str
    :param query_embedding: the embedding vector for vector-based search
    :type query_embedding: List[float]
    :param top_k: the number of top results to retrieve, defaults to 5
    :type top_k: int, optional
    :return: a list of serach results from OpenSearch
    :rtype: List[Dict[str, Any]]
    """
    client = get_opensearch_client()

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

    response = client.search(
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
    query_text: str,
    query_embedding: List[float],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Function that performs a hybrid search combining text-based and vector-based queries
    with respective BM25 and L2-ANN algorithms.

    :param query_text: the user text query for text-based search
    :type query_text: str
    :param query_embedding: the embedding vector for vector-based search
    :type query_embedding: List[float]
    :param top_k: the number of top results to retrieve, defaults to 5
    :type top_k: int, optional
    :return: a list of serach results from OpenSearch
    :rtype: List[Dict[str, Any]]
    """
    client = get_opensearch_client()

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

    response = client.search(
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
