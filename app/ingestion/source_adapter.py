"""
Abstract interface for product ingestion sources.

WHY THIS EXISTS:
Every source (CSV, WooCommerce, Shopify) has different mechanics but
must feed the same downstream pipeline. This interface guarantees they
all produce the same shape of output, so the mapper/normalizer/
canonicalizer don't care where products came from.
"""

from abc import ABC, abstractmethod


class SourceAdapter(ABC):
    """
    Base for all ingestion sources. Each concrete adapter fetches
    products in the source's native format — mapping/normalization
    happens downstream.
    """

    @abstractmethod
    def get_source_name(self) -> str:
        """Short identifier: 'csv', 'shopify', 'woocommerce'."""

    @abstractmethod
    def fetch_raw_products(self, store_id: str) -> list[dict]:
        """
        Fetch products in native format (dict per product).
        Downstream mapper normalizes column names, normalizer cleans
        values, converter builds Product objects.
        """

    @abstractmethod
    def get_source_columns(self, store_id: str) -> list[str]:
        """
        List of column/field names the source provides.
        Used by the deterministic mapper to build source → canonical map.
        """