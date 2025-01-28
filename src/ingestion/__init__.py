"""
Package containing loading and splitting functionalities for PDF files.

It is divied into two modules:

* loaders: contains classes to load PDF files as Langchain Document objects.
* splitters: contains classes to split Langchain Document objects into chunks.
"""

# Specify public objects

from .loaders import DocumentLoader
from .splitters import DocumentSplitter

__all__ = [
    "DocumentLoader",
    "DocumentSplitter",
]
