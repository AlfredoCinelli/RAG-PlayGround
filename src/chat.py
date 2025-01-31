"""
Module containig functions to create prompt and generate responses from Ollama.

The main functions of the module are:

* ensure_model_puller: Ensure that the chosen model is pulled and available locally.
* run_ollama_streaming: Run the specified model with streaming enabled.
* prompt_template: Build a prompt with context, conversation history, and user query.
* get_context_components: Get context components from search.
* generate_response_streaming: Generate a response from the specified model using streaming.
* aggregate_search_results: Aggregate results from lexical and vector searches.
* rerank_search_results: Rerank using cross-encoder results from lexical and vector searches.
* dump_chat: Save the chat history as csv file in a local directory.
* load_chat: Read chat history as csv file.
"""

# Import packages and modules

import os
from typing import Any, Dict, Iterator, List, Literal, Mapping

import ollama
import pandas as pd
import streamlit as st

from src.constants import ASSYMETRIC_EMBEDDING, MODEL_NAMES
from src.encoders import EmbeddingModel, RerankerModel
from src.logging import logger
from src.opensearch.retrieval import OpenSearchRetriever

# Define UDFs


@st.cache_resource(show_spinner=False)
def ensure_model_pulled(model: str) -> bool:
    """
    Function to ensure that the specified model is pulled and available locally.

    :param model: the name of the model to ensure is available
    :type model: str
    :return: True if the model is available or successfully pulled, False if an error occurs
    :rtype: bool
    """
    try:
        available_models = ollama.list()
        if model not in available_models:
            logger.info(f"Model {model} not found locally. Pulling the model...")
            ollama.pull(model)
            logger.info(f"Model {model} has been pulled and is now available locally.")
        else:
            logger.info(f"Model {model} is already available locally.")
    except ollama.ResponseError as e:
        logger.error(f"Error checking or pulling model: {e.error}")
        return False
    return True


def run_ollama_streaming(
    model: str,
    prompt: str,
    temperature: float,
) -> Iterator[Mapping[str, Any]] | None:
    """
    Function to call Ollama's Python library to run the chosen model with streaming enabled.

    :param model: name of the model (e.g., llama3.2:1b for Meta Llama 3.2 1B)
    :type model: str
    :param prompt: the prompt to send to the model
    :type prompt: str
    :param temperature: the response generation temperature
    :type temperature: float
    :return: a generator yielding response chunks as strings, or None if an error occurs
    :rtype: Optional[Iterator[Mapping[str, Any]]]
    """
    model_name = MODEL_NAMES[model]
    try:
        # Ensure the chosen model has been pulled
        ensure_model_pulled(model_name)
        # Now attempt to stream the response from the model
        logger.info(f"Streaming response from Ollama model: {model}.")
        stream = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"temperature": temperature},
        )
    except ollama.ResponseError as e:
        logger.error(f"Error during streaming: {e.error}")
        return None

    return stream


def prompt_template(
    query: str,
    context: str,
    history: List[Dict[str, str]],
    return_sources: bool = False,
) -> str:
    """
    Function to build the prompt with context, conversation history, and user query.

    :param query: the user's query
    :type query: str
    :param context: the context gathered from hybrid search
    :type context: str
    :param history: the conversation history
    :type history: List[Dict[str, str]]
    :return: the constructed prompt for Ollama model
    :rtype: str
    """
    prompt = "You are an assistant for question-answering tasks. "  # system prompt
    if (context) and (return_sources is True):
        logger.info("Generating prompt with context and sources.")
        prompt += (
            "Use the following context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Return also the source document name and page number with the answer. "
            "\nContext:\n" + context + "\n"
        )
    elif (context) and (return_sources is False):
        logger.info("Generating prompt with context but no sources.")
        prompt += (
            "Use the following context to answer the question."
            "If you don't know the answer, say that you don't know. "
            "\nContext:\n" + context + "\n"
        )
    else:
        logger.info("Generating prompt without context and sources.")
        prompt += "Answer questions to the best of your knowledge.\n"

    if history:
        logger.info("Including conversation history in the prompt.")
        prompt += "Conversation History:\n"
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            prompt += f"{role}: {content}\n"
        prompt += "\n"
    else:
        logger.info("No conversation history to include in the prompt.")

    prompt += f"User: {query}\nAssistant:"
    return prompt


def get_context_components(
    search_results: str,
) -> Dict[str, str | int]:
    """
    Function to extract the context components from the search results.

    :param search_results: single result from the context search
    :type search_results: str
    :return: context components extracted from the search results
    :rtype: Dict[str, Union[str, int]]
    """
    content_start = search_results.find("page_content='") + 14
    content_end = search_results.find("' metadata=")
    page_content = search_results[content_start:content_end]

    # Extract metadata between curly braces
    metadata_start = search_results.find("{")
    metadata_end = search_results.find("}") + 1
    metadata_str = search_results[metadata_start:metadata_end]
    metadata = eval(metadata_str)
    context_components = {
        "document_name": metadata.get("source").split("/")[-1].split(".")[0],
        "document_page_number": metadata.get("page") + 1,  # Page numbers start from 0
        "document_text": page_content,
    }

    return context_components


def generate_response_streaming(
    model: str,
    query: str,
    use_rag: bool,
    search_type: Literal["hybrid", "vector", "keyword", "reranking"],
    num_results: int,
    temperature: float,
    chat_history: List[Dict[str, str]] | None = None,
    return_sources: bool = False,
) -> Iterator[Mapping[str, Any]] | None:
    """
    Function to generate a chatbot response by incorporating conversation history
    and context based on the following search techniques:
        * lexical using Okapi BM25
        * vector using ANN
        * hybrid search using both lexical and vector search with RSF with custom weights.

    To know more about the search techniques, please refer to the OpenSearch config file (conf/*.json files).

    :param model: name of the model (e.g., llama3.2:1b for Meta Llama 3.2 1B)
    :type model: str
    :param query: the user's query
    :type query: str
    :param use_rag: whether to use RAG to get context
    :type use_rag: bool
    :param search_type: the type of search to use
    :type search_type: Literal["hybrid", "vector", "keyword"]
    :param num_results: the number of search results to include in the context
    :type num_results: int
    :param temperature: the temperature for the response generation
    :type temperature: float
    :param chat_history: the conversation history
    :type chat_history: Optional[List[Dict[str, str]]]
    :param return_sources: whether to return the source documents in the response
    :type return_sources: bool
    :return: generator yielding response chunks as strings, or None if an error occurs
    :rtype: Optional[Iterator[Mapping[str, Any]]]
    """
    chat_history = chat_history or []
    max_history_messages = 10
    history = chat_history[-max_history_messages:]
    context = ""

    if (use_rag is True) and (not search_type):
        logger.error("RAG search requires a search type.")
        raise ValueError("RAG search requires a search type.")

    if use_rag is True:
        logger.info("Performing RAG search.")
        retriever = OpenSearchRetriever()
        embedding = EmbeddingModel()
        # Apply the chosen search type
        match search_type:
            case "hybrid":
                logger.info("Performing hybrid search.")
                prefixed_query = f"passage: {query}" if ASSYMETRIC_EMBEDDING else f"{query}"
                query_embedding = embedding.embed_query(
                    prefixed_query,
                )
                search_results = retriever.hybrid_search(
                    query,
                    query_embedding,
                    top_k=num_results,
                )
                logger.info("Hybrid search completed.")
            case "vector":
                logger.info("Performing vector search.")
                query_embedding = embedding.embed_query(
                    query,
                )
                search_results = retriever.vector_search(
                    query,
                    query_embedding,
                    top_k=num_results,
                )
                logger.info("Vector search completed.")
            case "keyword":
                logger.info("Performing lexical search.")
                search_results = retriever.lexical_search(
                    query,
                    top_k=num_results,
                )
                logger.info("Lexical search completed.")
            case "reranking":
                logger.info("Performing reranking on lexical and vector search results.")
                query_embedding = embedding.embed_query(
                    query,
                )
                lexical_search_results = retriever.lexical_search(
                    query,
                    top_k=num_results,
                )
                vector_search_results = retriever.vector_search(
                    query,
                    query_embedding,
                    top_k=num_results,
                )
                search_results = RerankerModel().rerank_search_results(
                    vector_search_results=vector_search_results,
                    lexical_search_results=lexical_search_results,
                    query=query,
                    top_k=num_results,
                )

        # Collect text from search results
        for search_result in search_results:
            context += f"Document: {search_result.metadata.get('source')}, page: {search_result.metadata.get('page_number')}, content: \n{search_result.page_content}\n\n"  # type: ignore
        logger.info(
            f"Successfully extracted context components and assembled following context: {context}."
        )
    else:
        logger.info("No RAG search performed.")
    # Generate prompt using the prompt_template function
    prompt = prompt_template(query, context, history, return_sources)

    return run_ollama_streaming(model, prompt, temperature)


def dump_chat(
    chat_history: List[Dict[str, str]],
    directory: str = "chat_history",
    file_name: str = "chat.csv",
) -> None:
    """
    Function to dump the chat history as csv file in a given directory.

    :param chat_history: chat history from the conversation
    :type chat_history: Union[None, Dict[str, str]]
    :param directory: folder in which dump the chat history, defaults to 'chat_history'
    :type directory: str
    :param: file_name: name of the file representing the dumped chat
    :type file_name: str
    :return: dump the chat history as a csv file in the given directory
    :rtype: None
    """
    if chat_history:
        if not os.path.exists(directory):
            os.makedirs(directory)
        full_path = "/".join([directory, file_name])
        logger.info(f"Dumping the chat history in {full_path}")
        pd.DataFrame(chat_history).to_csv(full_path, index=False)
    else:
        logger.info("No chat history yet.")


@st.cache_data
def load_chat(
    chat_history: List[Dict[str, str]],
    directory: str = "chat_history",
    file_name: str = "chat.csv",
) -> bytes:
    """
    Function to load the chat as csv file.

    :param chat_history: history of the chat with the LLM
    :type chat_history: List[Dict[str, str]]
    :param directory: folder containing the chat file, defaults to "chat_history"
    :type directory: str, optional
    :param file_name: name of the chat csv file, defaults to "chat.csv"
    :type file_name: str, optional
    :return: chat history as csv (string file)
    :rtype: bytes
    """
    # Dump chat as csv
    dump_chat(chat_history, directory, file_name)
    full_path = "/".join([directory, file_name])

    # Read chat as csv
    chat_history_dumped = pd.read_csv(full_path)
    logger.info("Loaded chat history as csv file.")

    return chat_history_dumped.to_csv(index=False).encode("utf-8")
