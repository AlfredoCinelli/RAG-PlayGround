"""
Package containing OpenSearch-related functionality.

It is responsible for initializing the OpenSearch client.
"""

# Import packages and modules

from opensearchpy import OpenSearch
from typing_extensions import Self

from src.constants import (
    OPENSEARCH_HOST,
    OPENSEARCH_PORT,
)
from src.logging import logger

# Define Client class


class OpenSearchClient:
    """Parent class to initialize an OpenSearch client."""

    def __init__(
        self: Self,
    ) -> None:
        """
        OpenSearch client initialization.

        :param self: instance variable
        :type self: Self
        """
        self.client = OpenSearch(
            hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
            http_compress=True,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True,
        )
        logger.info("OpenSearch client initialized.")
