"""
Thin client for the Laptop-1 FastAPI backend.

Endpoints (see API_CONTRACT.md):
    GET  /health
    GET  /trips
    POST /predict   {"trip_id": "..."}

Every call returns (data, elapsed_ms): the UI shows the real measured
round-trip time rather than a generic spinner, which also doubles as a
quick way to show an evaluator that a result genuinely came from a live
network call to Laptop 1 and wasn't hardcoded.
"""

import time
import requests


class APIError(Exception):
    """Readable, frontend-facing API error."""


class PathSnapAPI:
    def __init__(self, base_url: str, timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, timeout=None, **kwargs):
        url = self.base_url + path
        start = time.perf_counter()
        try:
            response = requests.request(method, url, timeout=timeout or self.timeout, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            response.raise_for_status()
        except requests.RequestException as exc:
            raise APIError(
                f"Could not reach {url}. Check that FastAPI is running, the URL/port is "
                f"correct, and - for the two-laptop setup - that both laptops are on the "
                f"same network and the firewall allows the port. Details: {exc}"
            ) from exc
        try:
            return response.json(), elapsed_ms
        except ValueError as exc:
            raise APIError(f"{url} did not return valid JSON.") from exc

    def health(self, timeout: int = 4):
        """Returns (data, elapsed_ms)."""
        return self._request("GET", "/health", timeout=timeout)

    def get_trip_ids(self):
        """Returns a flat list of trip-id strings, however the backend shaped them."""
        data, _ = self._request("GET", "/trips")
        values = data if isinstance(data, list) else (
            data.get("trip_ids", data.get("trips", [])) if isinstance(data, dict) else []
        )
        trip_ids = []
        for item in values:
            value = item.get("trip_id", item.get("id")) if isinstance(item, dict) else item
            if value is not None:
                trip_ids.append(str(value))
        return trip_ids

    def predict(self, trip_id: str):
        """Returns (result_dict, elapsed_ms)."""
        data, elapsed_ms = self._request("POST", "/predict", json={"trip_id": str(trip_id)})
        if not isinstance(data, dict):
            raise APIError("/predict must return a JSON object.")
        return data, elapsed_ms
