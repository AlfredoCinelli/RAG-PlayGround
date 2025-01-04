"""
Module containig functions to ingest PDF files and extract chunks of text.

The main functions of the module are:

* extract_text_from_pdf: Extracts text from a PDF file using Langchain.
* text_chunking: Splits PDF into chunks with a specified overlap and length.
"""

# Import packages and modules

from typing import List, Literal

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_core.documents.base import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.logging import logger
from src.utils import clean_text

# Define literals

TEXT_SPLITTER = {
    "text": RecursiveCharacterTextSplitter,
    "length": CharacterTextSplitter,
}

# Define UDFs


def extract_text_from_pdf(
    file_path: str,
    to_string: bool = False,
) -> str | List[Document]:
    """
    Function to extract text from a PDF file.

    :param file_path: path to the PDF file
    :type file_path: str
    :param to_string: whether to get the PDF content as plain string, defaults to False
    :type to_string: bool, optional
    :return: PDF content as string or langchain Document objects
    :rtype: Union[str, List[Document]]
    """
    try:
        pdf_reader = PyPDFLoader(file_path, extract_images=True)
        logger.info(f"Opened PDF via PyPDFLoader for text extraction: {file_path}.")
        page_full_content = pdf_reader.load()
    except ValueError:
        pdf_reader = UnstructuredPDFLoader(file_path)  # type: ignore
        logger.info(
            f"Opened PDF via UnstructuredPDFLoader for text extraction: {file_path}."
        )
        page_full_content = pdf_reader.load()

    logger.info(f"Extracted text from PDF file: {file_path}.")
    text = (
        page_full_content
        if to_string is False
        else clean_text(
            "".join(
                [
                    page_full_content[page_num].page_content
                    for page_num in range(len(page_full_content))
                ]
            )
        )
    )

    return text


def text_chunking(
    text: str | List[Document],
    chunk_size: int,
    overlap: int,
    mode: Literal["text", "length"] = "text",
    tokenization: bool = False,
    to_string: bool = True,
) -> List[str | Document]:
    """
    Function to recursively chunk text given the chunk size and overlap.

    :param text: PDF content either as string or langchain Document objects
    :type text: str
    :param chunk_size: size of each chunk
    :type chunk_size: int
    :param overlap: number of characters to overlap between chunks
    :type overlap: int
    :param mode: whether to use a text-based or length-based splitter, defaults to "text"
    :type mode: Literal["text", "length"], optional
    :param tokenization: whether to use a tokenization count when splitting chunking, defaults to False
    :type tokenization: bool, optional
    :return: list of chunks from the given text
    :rtype: List[Union[str, Document]]
    """
    text_splitter = (
        TEXT_SPLITTER.get(mode)(  # type: ignore
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        if tokenization is False
        else TEXT_SPLITTER.get(mode).from_tiktoken_encoder(  # type: ignore
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
    )
    logger.info(
        f"Chunking text into chunks of size {chunk_size} and overlap {overlap}."
    )
    chunks = text_splitter.split_documents(text)
    logger.info("Successfully chunked text.")
    cleaned_chunks = [
        Document(
            metadata=chunk.metadata,
            page_content=clean_text(chunk.page_content, print_log=False),
        )
        for chunk in chunks
    ]
    logger.info("Text cleaned.")
    return (
        cleaned_chunks  # type: ignore
        if to_string is False
        else [chunk.page_content for chunk in cleaned_chunks]  # type: ignore
    )
