"""
Module containig classes to split Langchain Document objects into smaller chunks.

The main classes of the module are:

* SplitterStrategy: abstract class defining the strategy to build different splitters.
* SplittierStrategyFactory: class to create different splitters based on the given strategy.
* DocumentSplitter: abstract class defining the common interface for document splitters.
* CharacterSplitterStrategy: concrete class of DocumentSplitter to create a plain text splitter.
* RecursiveSplitterStrategy: concrete class of DocumentSplitter to create a recursive text splitter.
"""

# Import packages and modules
from abc import ABC, abstractmethod
from typing import List, Literal, Type

from langchain_core.documents.base import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)
from typing_extensions import Self

from src.logging import logger

# Define classes


class SplitterStrategy(ABC):
    """Abstract class defining the common interface for document splitting strategies."""

    def __init__(
        self: Self,
        chunk_size: int,
        overlap: int,
        tokenization: bool,
    ) -> None:
        """
        Initialize the document splitter.

        :param self: instance of the class
        :type self: Self
        :param chunk_size: number characters/tokens per chunk
        :type chunk_size: int
        :param overlap: number of characters/tokens to overlap between chunks
        :type overlap: int
        :param tokenization: whether to measure the chunk length via tokens
        :type tokenization: bool
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenization = tokenization
        self.splitter = self._create_splitter()

    @abstractmethod
    def _create_splitter(self: Self) -> TextSplitter:
        """
        Abstract private method to create an instance of a text splitter.

        :return: instance of LangChain TextSplitter
        :rtype: TextSplitter
        """

    @abstractmethod
    def split(self: Self, document: List[Document]) -> List[Document]:
        """
        Abstract method to be implemented by concrete subclasses.
        The method should return a list of Document objects.

        :param self: instance of the class
        :type self: Self
        :return: list of Langchain Document objects
        :rtype: List[Document]
        """


class RecursiveSplitterStrategy(SplitterStrategy):
    """Concrete class of SplitterStrategy to chunk a document in a recursive fashion."""

    def _create_splitter(self: Self) -> TextSplitter:
        """Private method to create a RecursiveCharacterTextSplitter."""
        return (
            RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
            )
            if self.tokenization is True
            else RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
            )
        )

    def split(self: Self, document: List[Document]) -> List[Document]:
        """
        Method to split a PDF file using Langchain RecursiveCharacterTextSplitter.

        :param self: instance of the class
        :type self: Self
        :param documents: list of Langchain document objects
        :type documents: List[Document]
        :return: list of Langchain Document objects
        :rtype: List[Document]
        """
        return self.splitter.split_documents(document)


class CharacterSplitterStrategy(SplitterStrategy):
    """Concrete class of SplitterStrategy to chunk a document in a plain fashion."""

    def _create_splitter(self: Self) -> TextSplitter:
        """Private method to create a CharacterTextSplitter."""
        return (
            CharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
            )
            if self.tokenization is True
            else CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
            )
        )

    def split(self: Self, document: List[Document]) -> List[Document]:
        """
        Method to split a PDF file using Langchain CharacterTextSplitter.

        :param self: instance of the class
        :type self: Self
        :param documents: list of Langchain document objects
        :type documents: List[Document]
        :return: list of Langchain Document objects
        :rtype: List[Document]
        """
        return self.splitter.split_documents(document)


class SplittierStrategyFactory:
    """Factory for creating splitting strategies."""

    _strategies = {"recursive": RecursiveSplitterStrategy, "character": CharacterSplitterStrategy}

    @classmethod
    def create_splitter(
        cls: Type["SplittierStrategyFactory"],
        mode: Literal["character", "recursive"],
        chunk_size: int,
        overlap: int,
        tokenization: bool = False,
    ) -> SplitterStrategy:
        """
        Create a splitting strategy instance.

        :param mode: which splitting strategy to use
        :type mode: Literal["character", "recursive"]
        :param chunk_size: number characters/tokens per chunk
        :type chunk_size: int
        :param overlap: number of characters/tokens to overlap between chunks
        :type overlap: int
        :param tokenization: whether to measure the chunk length via tokens
        :type tokenization: bool
        """
        splitter = cls._strategies.get(mode)
        if not splitter:
            raise ValueError(f"Unsupported strategy type: {mode}")
        return splitter(chunk_size, overlap, tokenization)


class DocumentSplitter:
    """Context class to create an instance of the splitter from the strategy."""

    def __init__(
        self: Self,
        chunk_size: int,
        overlap: int,
        tokenization: bool,
        mode: Literal["character", "recursive"],
    ) -> None:
        """
        Constructor of the DocumentSplitter.
        Based on the mode it calls the SplitterStrategyFactory
        to create an instance of the different available
        document splitters.

        :param mode: which splitting strategy to use
        :type mode: Literal["character", "recursive"]
        :param chunk_size: number characters/tokens per chunk
        :type chunk_size: int
        :param overlap: number of characters/tokens to overlap between chunks
        :type overlap: int
        :param tokenization: whether to measure the chunk length via tokens
        :type tokenization: bool
        """
        splitter = SplittierStrategyFactory.create_splitter(mode, chunk_size, overlap, tokenization)
        if splitter is None:
            msg = f"Invalid splitting mode: {mode}"
            logger.error(msg)
            raise ValueError(msg)
        else:
            self.splitter = splitter

    def split(self: Self, document: List[Document]) -> List[Document]:
        """Method calling the split to split the documents."""
        return self.splitter.split(document)
