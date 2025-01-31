"""
Module containig class to generate embeddings from text.

The main classes of the module are:

* RerankerModel: Class to rank documents against a query (via cross-encoders)

The main functions of the module are:

* aggregate_search_results: Aggregates results from different searches for reranking.
* reranke_search_results: Takes results from semantic and lexical search, the query and reranks outputting the top k.
"""

# Import packages and modules

import typing
from typing import Any, Dict, List

import streamlit as st
from langchain_core.documents.base import Document
from sentence_transformers.cross_encoder import CrossEncoder
from typing_extensions import Self

from src.constants import RERANKER_MODEL_PATH
from src.logging import logger


# Define UDFs
class RerankerModel:
    """Class representing a Reranker model."""

    def __init__(
        self: Self,
        model_name: str = RERANKER_MODEL_PATH,
    ) -> None:
        """
        Initialize the RerankerModel.

        :param self: instance variable
        :type self: Self
        :param model_name: the name of the model to use, defaults to RERANKER_MODEL_PATH
        :type model_name: str, optional
        """
        self.model_name = model_name
        self.model = self.get_model()

    @st.cache_resource(show_spinner=False)
    def get_model(
        _self: Self,  # noqa: N805
    ) -> CrossEncoder:
        """
        Method to get the reranker model and cache it accross sessions and reruns.

        :param _self: instance variable
        :type _self: Self
        :return: instance of HuggingFace CrossEncoder
        :rtype: CrossEncoder
        """
        return CrossEncoder(RERANKER_MODEL_PATH)

    @typing.no_type_check
    def aggregate_search_results(
        self: Self,
        lexical_search_results: List[Dict[str, Any]],
        vector_search_results: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Function to aggregate results from lexical and vector search.
        It gives back a dictionary with lexical and vector search results
        as values and extracted text as key.

        :param lexical_search_results: results from lexical search
        :type lexical_search_results: List[Dict[str, Any]]
        :param vector_search_results: results from vector search
        :type vector_search_results: List[Dict[str, Any]]
        :return: dictionary with lexical and vector search results as values and extracted text as key
        :rtype: Dict[str, Dict[str, Any]]
        """
        aggregated_search_results = {}
        logger.info("Aggregating results from lexical and vector search.")

        for lexical_result, vector_result in zip(
            lexical_search_results, vector_search_results, strict=False
        ):
            (
                aggregated_search_results[lexical_result.page_content],
                aggregated_search_results[vector_result.page_content],
            ) = (
                lexical_result,
                vector_result,
            )

        logger.info("Successfully aggregated results from lexical and vector search.")

        return aggregated_search_results

    @typing.no_type_check
    def rerank_search_results(
        self: Self,
        vector_search_results: List[Dict[str, Any]],
        lexical_search_results: List[Dict[str, Any]],
        query: str,
        top_k: int = 5,
    ) -> List[Document]:
        """
        Function that applies a cross-encoder to rerank search results.
        The aim of the function is to rerank the results from lexical and vector search.

        :param vector_search_results: results from vector search
        :type vector_search_results: List[Dict[str, Any]]
        :param lexical_search_results: results from lexical search
        :type lexical_search_results: List[Dict[str, Any]]
        :param query: user query
        :type query: str
        :param top_k: maximum number of results to retrieve, defaults to 5
        :type top_k: int, optional
        :return: reranked results from both lexical and vector search results as Langchain Document objects
        :rtype: List[Document]
        """
        logger.info("Aggregating results from lexical and vector search.")
        aggregated_search_results = self.aggregate_search_results(
            lexical_search_results,
            vector_search_results,
        )
        logger.info(f"Using model {RERANKER_MODEL_PATH} as cross-encoder to rerank search results.")
        corpus = list(aggregated_search_results.keys())

        ranks = self.model.rank(
            query=query,
            documents=corpus,
            top_k=top_k,
        )
        logger.info(f"Successfully reranked search results with {top_k} results.")

        reranked_corpus = [corpus[rank["corpus_id"]] for rank in ranks]
        reranked_search_results = {
            key: aggregated_search_results[key]
            for key in reranked_corpus
            if key in aggregated_search_results
        }
        return list(reranked_search_results.values())
