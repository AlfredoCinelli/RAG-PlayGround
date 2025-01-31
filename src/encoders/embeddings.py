"""
Module containig class to generate embeddings from text.

The main classes of the module are:

* EmbeddingModel: Class to generate embeddings from text

The main functions of the module are:

* embed_documents: Embedd list of documents
* embed_query: Embed query
"""

# Import packages and modules

import typing
from typing import List

import streamlit as st
from langchain_core.documents.base import Document
from langchain_huggingface import HuggingFaceEmbeddings
from typing_extensions import Self

from src.constants import EMBEDDING_MODEL_PATH
from src.utils import check_string_list, logger

# Define UDFs


@typing.no_type_check
def extract_documents_content(
    documents: List[Document] | Document | List[str] | str,
) -> List[str]:
    """
    Extracts the content from a list of documents or a single document.

    :param document: the documents from which extract the content,
        It can be:
        * string representing the content already
        * single Document object
        * list of Document objects
        * list of strings representing the content of the documents
    :type document: Union[List[Document], Document, List[str], str]
    :return: the content of the documents
    :rtype: List[str]
    """
    documents = [documents] if isinstance(documents, Document | str) else documents
    return (
        documents
        if check_string_list(documents) or isinstance(documents, str)
        else [document.page_content for document in documents]
    )


class EmbeddingModel:
    """Class representing an embedding model."""

    def __init__(
        self: Self,
        model_name: str = EMBEDDING_MODEL_PATH,
    ) -> None:
        """
        Initialize the EmbeddingModel.

        :param self: instance variable
        :type self: Self
        :param model_name: the name of the model to use, defaults to EMBEDDING_MODEL_PATH
        :type model_name: str, optional
        """
        self.model_name = model_name
        self.embedding_model = self.get_model()

    @st.cache_resource(show_spinner=False)
    def get_model(
        _self: Self,  # noqa: N805
    ) -> HuggingFaceEmbeddings:
        """
        Method to get the embedding model and cache it accross sessions and reruns.

        :param _self: instance variable
        :type _self: Self
        :return: instance of the Langchain embedding model
        :rtype: HuggingFaceEmbeddings
        """
        return HuggingFaceEmbeddings(
            model_name=_self.model_name,
            model_kwargs={"device": "cpu"},
            show_progress=True,
        )

    def embed_documents(
        self: Self,
        documents: List[str],
    ) -> List[List[float]]:
        """
        Method to generate embeddings from documents.

        :param self: instance variable
        :type self: Self
        :param documents: list of documents content
        :type documents: List[str]
        :return: list of documents content embeddings (embeddings dimension depends upon the chosen model)
        :rtype: List[np.typing.NDArray[np.float32]]
        """
        documents_content = extract_documents_content(documents)
        logger.info(
            f"Generated embeddings for {len(documents) if isinstance(documents, list) else 1} text chunks."
        )

        return self.embedding_model.embed_documents(documents_content)

    def embed_query(
        self: Self,
        query: str,
    ) -> List[float]:
        """
        Method to generate embeddings from a query.

        :param self: instance variable
        :type self: Self
        :param query: the query to generate embeddings from
        :type query: str
        :return: query embeddings (embeddings dimension depends upon the chosen model)
        :rtype: List[np.typing.NDArray[np.float32]]
        """
        logger.info("Generated embeddings for query.")
        return self.embedding_model.embed_query(query)
