"""
CSV/Excel adapter. Parses uploaded file into raw dict rows.
"""

import io
import pandas as pd

from app.ingestion.source_adapter import SourceAdapter


class CSVAdapter(SourceAdapter):
    """
    Parses CSV or Excel file bytes into raw product dicts.
    Column names preserved as-is; mapping happens downstream.
    """

    def __init__(self, file_bytes: bytes, filename: str):
        self._file_bytes = file_bytes
        self._filename = filename.lower()
        self._df: pd.DataFrame | None = None

    def get_source_name(self) -> str:
        return "csv"

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        buf = io.BytesIO(self._file_bytes)
        if self._filename.endswith((".xlsx", ".xls")):
            self._df = pd.read_excel(buf)
        else:
            self._df = pd.read_csv(buf)
        # Convert NaN to None for clean dicts
        self._df = self._df.where(pd.notna(self._df), None)
        return self._df

    def get_source_columns(self, store_id: str) -> list[str]:
        return [str(c) for c in self._load().columns.tolist()]

    def fetch_raw_products(self, store_id: str) -> list[dict]:
        return self._load().to_dict(orient="records")
    