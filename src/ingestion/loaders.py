"""
Module containig classes to ingest PDF files as Langchain Document objects.

The main classes of the module are:

* LoaderStrategy: abstract class defining the strategy to build different loaders.
* LoaderStrategyFactory: class to create different loaders based on the given strategy.
* DocumentLoader: context class building the chosen loader.
* DocumentFilterMixIn: mixin class to filter Langchain Documents to keep only the desired content.
* PyPDFStrategy: concrete class of DocumentLoader to load a PDF file using PyPDF.
* UnstructuredStrategy: concrete class of DocumentLoader to load a PDF file using Unstructured.
"""

# Import packages and modules
from abc import ABC, abstractmethod
from typing import List, Literal, Type

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPDFLoader,
)
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents.base import Document
from typing_extensions import Self

from src.logging import logger

# Define classes


class LoaderStrategy(ABC):
    """Abstract class defining the common interface for document loader strategies."""

    def __init__(
        self: Self,
        file_path: str,
    ) -> None:
        """
        Initialize the document loader.

        param: self: instance of the class
        :type self: Self
        :param file_path: path to the PDF file
        :type file_path: str
        """
        self.file_path = file_path
        self.loader = self._create_loader()

    @abstractmethod
    def _create_loader(self: Self) -> BaseLoader:
        """
        Abstract private method to create an instance of a loader.

        :return: instance of LangChain BaseLoader
        :rtype: BaseLoader
        """

    @abstractmethod
    def load(self: Self) -> List[Document]:
        """
        Abstract method to be implemented by concrete subclasses.
        The method should return a list of Document objects.

        :param self: instance of the class
        :type self: Self
        :return: list of Langchain Document objects
        :rtype: List[Document]
        """


class DocumentFilterMixIn:
    """Mixin class to filter Langchain Documents to keep only the desired content."""

    @staticmethod
    def _filter_document(
        documents: List[Document],
    ) -> List[Document]:
        """
        Method to reformat Document objects from LangChain.
        The method get the text in the page_content and
        keep only the metadata that are relevant.

        :param documents: list of LangChain Document objects
        :type documents: List[Document]
        :return: list of cleaned and filtered LangChain Document objects
        :rtype: List[Document]
        """
        return [
            Document(
                page_content=document.page_content,
                metadata={
                    key: val
                    for key, val in zip(
                        document.metadata.keys(), document.metadata.values(), strict=True
                    )
                    if key in ["source", "page", "page_number"]
                },
            )
            for document in documents
        ]


class PyPDFStrategy(
    LoaderStrategy,
    DocumentFilterMixIn,
):
    """Concrete class of DocumentLoader to load a PDF file using PyPDF."""

    def _create_loader(
        self: Self,
    ) -> BaseLoader:
        """Private method to create a PyPDFLoader."""
        return PyPDFLoader(
            file_path=self.file_path,
            extract_images=True,
        )

    def load(self: Self) -> List[Document]:
        """
        Method to load a PDF file using Langchain PyPDF.

        :param self: instance of the class
        :type self: Self
        :return: list of Langchain Document objects
        :rtype: List[Document]
        """
        return self.loader.load()


class UnstructuredStrategy(
    LoaderStrategy,
    DocumentFilterMixIn,
):
    """Concrete class of DocumentLoader to load a PDF file using Unstructured."""

    def _create_loader(self: Self) -> BaseLoader:
        """Private method to create an UnstructuredPDFLoader."""
        return UnstructuredPDFLoader(
            file_path=self.file_path,
            mode="elements",
            strategy="hi_res",
        )

    def load(self: Self) -> List[Document]:
        """
        Method to load a PDF file using Langchain Unstructured loadr.

        :param self: instance of the class
        :type self: Self
        :return: list of Langchain Document objects
        :rtype: List[Document]
        """
        return self.loader.load()


class LoaderStrategyFactory:
    """Factory class to create instances of DocumentLoader subclasses."""

    # Dictionary to lookyp available document loaders
    _loaders = {
        "pypdf": PyPDFStrategy,
        "unstructured": UnstructuredStrategy,
    }

    @classmethod
    def create_loader(
        cls: Type["LoaderStrategyFactory"],
        mode: Literal["pypdf", "unstructured"],
        file_path: str,
    ) -> LoaderStrategy:
        """
        Method to create an instance of a DocumentLoader subclass.

        :param mode: the mode to use for loading the PDF
        :type mode: Literal["pypdf", "unstructured"]
        :param file_path: path to the PDF file
        :type file_path: str
        :raises ValueError: raise error when the given loader does not exist
        :return: instance of the DocumentLoader subclass
        :rtype: DocumentLoader
        """
        loader = cls._loaders.get(mode, None)
        if loader is None:
            msg = f"Unsupported loader mode: {mode}"
            logger.error(msg)
            raise ValueError(msg)
        else:
            return loader(file_path=file_path)


class DocumentLoader:
    """Context class to create an instance of the loaders from the strategy."""

    def __init__(
        self: Self,
        mode: Literal["pypdf", "unstructured"],
        file_path: str,
    ) -> None:
        """
        Constructor of the DocumentLoader.
        Based on the mode it calls the LoaderStrategyFactory
        to create an instance of the different available
        document loaders.

        :param mode: document loader strategy
        :type mode: Literal["pypdf", "unstructured"]
        :param file_path: path to the pdf file to be loaded
        :type file_path: str
        """
        loader = LoaderStrategyFactory.create_loader(
            mode=mode,
            file_path=file_path,
        )
        if loader is None:
            msg = f"Invalid splitting mode: {mode}"
            logger.error(msg)
            raise ValueError(msg)
        else:
            self.loader = loader

    def load(self: Self) -> List[Document]:
        """Method calling the loader to load the documents."""
        return self.loader.load()
