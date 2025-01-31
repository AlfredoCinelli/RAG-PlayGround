"""
Package containing documents embeddings generators and documents reranker.

It is divied into two modules:

* embeddigs: module containing classes to generate text embeddings
"""

# Specify public objects

from .embeddings import EmbeddingModel
from .reranker import RerankerModel

__all__ = [
    "EmbeddingModel",
    "RerankerModel",
]
