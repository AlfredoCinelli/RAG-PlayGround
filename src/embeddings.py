"""
Module containing functions to generate embeddings for text.

The main functions of the module are:

* get_embedding_model: Loads and caches (via Streamlit decorator) the embedding model.
* generate_embeddings: Generates embeddings for a list of texts using the embedding model.
"""

# Import packages and modules

import logging
from typing import List, Union

import numpy as np
import streamlit as st
from langchain_core.documents.base import Document
from sentence_transformers import SentenceTransformer

from src.constants import EMBEDDING_MODEL_PATH
from src.utils import check_string_list, setup_logging

# Initialize logger
setup_logging()  # Configures logging for the application
logger = logging.getLogger(__name__)

# Define UDFs


@st.cache_resource(show_spinner=False)
def get_embedding_model() -> SentenceTransformer:
    """
    Function to load an embedding model from SentenceTransformer,
    based on a chosen model via EMBEDDING_MODEL_PATH and cachinig it
    using Streamlit's caching resource decorator.

    :return: the loaded embedding model
    :rtype: SentenceTransformer
    """

    logger.info(f"Loading embedding model from path: {EMBEDDING_MODEL_PATH}")
    return SentenceTransformer(EMBEDDING_MODEL_PATH)


def generate_embeddings(
    chunks: Union[List[str], str, List[Document], Document],
) -> List[np.typing.NDArray[np.float32]]:
    """
    Function to generate embeddings from a list of chunked text via
    SentenceTransformer loaded embedding model.

    :param chunks: list of text chunks
    :type chunks: Union[List[str], str]
    :return: list of embeddings as numpy arrays for each chunk
    :rtype: List[np.ndarray]
    """
    chunks = (
        chunks
        if (check_string_list(chunks) or isinstance(chunks, str))  # type: ignore
        else [chunk.page_content for chunk in chunks]  # type: ignore
    )
    model = get_embedding_model()
    embeddings = model.encode(chunks)  # type: ignore
    logger.info(
        f"Generated embeddings for {len(chunks) if isinstance(chunks, list) else 1} text chunks."
    )
    return list(embeddings)
