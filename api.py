"""
api.py — Web of Science Expanded API client.

Handles authentication, pagination, retry logic, and date-range queries.
Returns raw record dicts for downstream parsing in core.py.
"""

import logging
import time
from typing import Generator

import requests

logger = logging.getLogger(__name__)


class WoSAPIError(Exception):
    """Raised when the WoS API returns an unrecoverable error."""


class WoSClient:
    """Thin client for the WoS Expanded API (JSON)."""

    def __init__(self, api_key: str, cfg: dict):
        self.api_key = api_key
        self.base_url = cfg["api"]["base_url"]
        self.page_size = int(cfg["api"]["page_size"])
        self.max_retries = int(cfg["api"]["max_retries"])
        self.backoff_base = float(cfg["api"]["backoff_base"])
        self.session = requests.Session()
        self.session.headers.update(
            {"X-ApiKey": self.api_key, "Accept": "application/json"}
        )

    def query_date_range(
        self,
        start_date: str,
        end_date: str,
        database: str = "WOS",
        extra_query: str = "",
    ) -> Generator[dict, None, None]:
        """Yield raw WoS record dicts for a publication date range.

        Args:
            start_date: Inclusive start in YYYY-MM-DD format.
            end_date:   Inclusive end in YYYY-MM-DD format.
            database:   WoS database code (default ``"WOS"``).
            extra_query: Additional query terms ANDed to the date filter.
        """
        start_year = start_date[:4]
        end_year = end_date[:4]
        if start_year == end_year:
            date_part = f"PY={start_year}"
        else:
            date_part = f"PY=({start_year}-{end_year})"

        query = f"OG=(Medical University Varna) AND {date_part}"
        if extra_query:
            query = f"({query}) AND ({extra_query})"

        logger.info("WoS query: %s", query)

        first_record = 1
        total_fetched = 0

        while True:
            batch = self._fetch_page(query, database, first_record)
            record_list = self._extract_record_list(batch)

            if not record_list:
                logger.info("No more records. Total fetched: %d", total_fetched)
                break

            for rec in record_list:
                yield rec

            total_fetched += len(record_list)
            logger.info("Fetched %d records (page starting at %d)", len(record_list), first_record)

            if len(record_list) < self.page_size:
                break

            first_record += self.page_size

    def _fetch_page(self, query: str, database: str, first_record: int) -> dict:
        """Fetch one page with exponential-backoff retry."""
        params = {
            "databaseId": database,
            "usrQuery": query,
            "count": self.page_size,
            "firstRecord": first_record,
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(self.base_url, params=params, timeout=30)

                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = self.backoff_base ** attempt
                    logger.warning("HTTP %d. Waiting %.1fs (attempt %d)", resp.status_code, wait, attempt)
                    time.sleep(wait)
                    continue

                raise WoSAPIError(f"API error {resp.status_code}: {resp.text[:300]}")

            except requests.exceptions.ConnectionError as exc:
                wait = self.backoff_base ** attempt
                logger.warning("Connection error: %s. Retrying in %.1fs", exc, wait)
                time.sleep(wait)

        raise WoSAPIError(
            f"Failed after {self.max_retries} attempts at firstRecord={first_record}"
        )

    @staticmethod
    def _extract_record_list(batch: dict) -> list:
        """Extract the list of records from a raw API response.

        Confirmed JSON path from export.json: top-level has 'Data' key,
        then Data → Records → records → REC (list).
        Falls back defensively to other known variants.
        """
        if not batch or not isinstance(batch, dict):
            return []

        # Primary: Data → Records → records → REC
        try:
            recs = batch["Data"]["Records"]["records"]["REC"]
            return recs if isinstance(recs, list) else [recs]
        except (KeyError, TypeError):
            pass

        # Secondary: Records → records → REC  (some endpoints omit 'Data')
        try:
            recs = batch["Records"]["records"]["REC"]
            return recs if isinstance(recs, list) else [recs]
        except (KeyError, TypeError):
            pass

        # Tertiary: walk tree looking for REC
        def _find_rec(obj, depth=0):
            if depth > 6 or not isinstance(obj, dict):
                return None
            for k, v in obj.items():
                if k == "REC":
                    return v if isinstance(v, list) else [v]
                found = _find_rec(v, depth + 1)
                if found is not None:
                    return found
            return None

        found = _find_rec(batch)
        if found:
            logger.warning("Used fallback REC search — check API response structure")
            return found

        logger.error("Could not find records. Top-level keys: %s", list(batch.keys()))
        return []


def validate_api_key(api_key: str, cfg: dict) -> bool:
    """Quick connectivity check. Returns True if the API key is accepted."""
    client = WoSClient(api_key, cfg)
    try:
        resp = client.session.get(
            cfg["api"]["base_url"],
            params={"databaseId": "WOS", "usrQuery": "TS=test", "count": 1, "firstRecord": 1},
            timeout=10,
        )
        return resp.status_code in (200, 204)
    except Exception:
        return False
