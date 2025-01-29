"""
Module defining a class to perform different types of search on an OpenSearch index.

The main class is OpenSearchRetriever with main methods:

* lexical_search: performs a lexical search on an OpenSearch index
* vector_search: performs a vector search on an OpenSearch index
* hybrid_search: performs a hybrid search on an OpenSearch index
"""

# Import packages and modules

import typing
from typing import Any, Dict, List

from langchain_core.documents import Document
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

    @staticmethod
    @typing.no_type_check
    def search_to_document(
        search_result: Dict[str, Any],
    ) -> Document:
        """
        Function to extract Document object from the search results.

        :param search_results: single result from the context search
        :type search_results: str
        :return: Langchain Document object with page_content, metadata (i.e., source and page number)
        :rtype: Document
        """
        search_result = search_result.get("_source").get("text")
        content_start = search_result.find("page_content='") + 14
        content_end = search_result.find("' metadata=")
        page_content = search_result[content_start:content_end]

        # Extract metadata between curly braces
        metadata_start = search_result.find("{")
        metadata_end = search_result.find("}") + 1
        metadata_str = search_result[metadata_start:metadata_end]
        metadata = eval(metadata_str)
        document = Document(
            page_content=page_content,
            metadata={
                "source": metadata.get("source").split("/")[-1].split(".")[0],
                "page_number": metadata.get("page") + 1,  # Page numbers start from 0
            },
        )

        return document

    def lexical_search(
        self: Self,
        query_text: str,
        top_k: int = 5,
    ) -> List[Document]:
        """
        Method that performs a lexical search using BM25.

        :param self: instance variable
        :type self: Self
        :param query_text: the user text query for text-based search
        :type query_text: str
        :param top_k: the number of search results to return, defaults to 5
        :type top_k: int, optional
        :return: a list of serach results from OpenSearch formatted as Langchain Document objects
        :rtype: List[Document]
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
        logger.info("Converting search results into Langchain document objects.")
        documents = [self.search_to_document(search_result=search_result) for search_result in hits]
        return documents

    def vector_search(
        self: Self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Document]:
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
        :return: a list of serach results from OpenSearch formatted as Langchain Document objects
        :rtype: List[Document]
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
        logger.info("Converting search results into Langchain document objects.")
        documents = [self.search_to_document(search_result=search_result) for search_result in hits]
        return documents

    def hybrid_search(
        self: Self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Document]:
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
        :return: a list of serach results from OpenSearch formatted as Langchain Document objects
        :rtype: List[Document]
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
        logger.info("Converting search results into Langchain document objects.")
        documents = [self.search_to_document(search_result=search_result) for search_result in hits]
        return documents
