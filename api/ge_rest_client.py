"""
OSRS Real-time Prices API Client

A compliant client for the prices.runescape.wiki API that follows all User-Agent
requirements and rate limiting guidelines.

API Documentation: https://oldschool.runescape.wiki/w/RuneScape:Real-time_Prices
"""

import requests
import time
import logging
from typing import Dict, List, Any, Optional, Union
from functools import lru_cache

# Set up logger
logger = logging.getLogger("GrandExchangeClient")


class GrandExchangeAPIError(Exception):
    """Exception raised for API errors."""
    pass


class RateLimitError(GrandExchangeAPIError):
    """Exception raised when rate limited."""
    pass


class GrandExchangeClient:
    """
    Client for the OSRS Real-time Prices API.

    Follows all API guidelines including:
    - Descriptive User-Agent header (REQUIRED)
    - Reasonable request frequency
    - Proper error handling
    """

    BASE_URL = "https://prices.runescape.wiki/api/v1/osrs"

    # Valid timesteps for timeseries endpoint
    VALID_TIMESTEPS = ["5m", "1h", "6h", "24h"]

    def __init__(
        self,
        user_agent: str = None,
        contact_email: str = None,
        project_name: str = "PPOFlipper",
        cache_size: int = 128,
        max_retries: int = 3,
        backoff_factor: float = 1.0
    ):
        """
        Initialize the Grand Exchange API client.

        Args:
            user_agent: Full custom User-Agent string. If not provided, one will be
                       constructed from project_name and contact_email.
            contact_email: Contact email for the User-Agent (required if user_agent not set)
            project_name: Project name for the User-Agent
            cache_size: Size of the LRU cache for API responses
            max_retries: Maximum number of retries for failed requests
            backoff_factor: Backoff multiplier for retries
        """
        # Build User-Agent header - REQUIRED by the API
        if user_agent:
            self._user_agent = user_agent
        elif contact_email:
            self._user_agent = f"{project_name} - OSRS GE Flipper Training Bot ({contact_email})"
        else:
            raise ValueError(
                "User-Agent is REQUIRED by the API. Provide either 'user_agent' or 'contact_email'. "
                "Generic user agents (python-requests, curl, etc.) are blocked. "
                "See: https://oldschool.runescape.wiki/w/RuneScape:Real-time_Prices"
            )

        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": self._user_agent,
            "Accept": "application/json"
        })

        self._max_retries = max_retries
        self._backoff_factor = backoff_factor
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms minimum between requests

        # ID/name mappings
        self._name_to_id_map: Dict[str, str] = {}
        self._id_to_name_map: Dict[str, str] = {}
        self._item_metadata: Dict[str, Dict] = {}

        logger.info(f"GrandExchangeClient initialized with User-Agent: {self._user_agent}")

    def _request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Make a request to the API with retry logic.

        Args:
            endpoint: API endpoint (e.g., "/latest", "/mapping")
            params: Query parameters

        Returns:
            JSON response as dictionary

        Raises:
            GrandExchangeAPIError: On API errors
            RateLimitError: When rate limited
        """
        url = f"{self.BASE_URL}{endpoint}"

        for attempt in range(self._max_retries):
            # Rate limiting - ensure minimum interval between requests
            elapsed = time.time() - self._last_request_time
            if elapsed < self._min_request_interval:
                time.sleep(self._min_request_interval - elapsed)

            try:
                response = self._session.get(url, params=params, timeout=30)
                self._last_request_time = time.time()

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limited
                    wait_time = (attempt + 1) * self._backoff_factor * 5
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 403:
                    raise GrandExchangeAPIError(
                        f"403 Forbidden - Your User-Agent may be blocked. "
                        f"Current: {self._user_agent}"
                    )
                else:
                    logger.warning(
                        f"Request failed with status {response.status_code}: {response.text}"
                    )

            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout on attempt {attempt + 1}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error on attempt {attempt + 1}: {e}")

            # Exponential backoff
            if attempt < self._max_retries - 1:
                wait_time = (attempt + 1) * self._backoff_factor
                time.sleep(wait_time)

        raise GrandExchangeAPIError(f"Failed to fetch {endpoint} after {self._max_retries} attempts")

    def get_mapping(self) -> List[Dict]:
        """
        Fetch item mapping data (IDs, names, limits, alch values).

        Returns:
            List of item metadata dictionaries with keys:
            - id: Item ID
            - name: Item name
            - examine: Examine text
            - members: Whether members-only
            - lowalch: Low alch value
            - highalch: High alch value
            - limit: GE buy limit
            - value: Store value
        """
        response = self._request("/mapping")

        # Build lookup maps
        for item in response:
            item_id = str(item.get("id"))
            item_name = item.get("name", "")

            self._id_to_name_map[item_id] = item_name
            self._name_to_id_map[item_name] = item_id
            self._item_metadata[item_id] = item

        logger.info(f"Loaded mapping for {len(response)} items")
        return response

    def get_latest(self, item_id: Optional[int] = None) -> Dict:
        """
        Fetch latest high/low prices for items.

        Args:
            item_id: Optional specific item ID. If None, returns all items.

        Returns:
            Dictionary with item IDs as keys and price data as values:
            {
                "item_id": {
                    "high": int,      # Highest instant-buy price
                    "highTime": int,  # Unix timestamp
                    "low": int,       # Lowest instant-sell price
                    "lowTime": int    # Unix timestamp
                }
            }
        """
        params = {"id": item_id} if item_id else None
        response = self._request("/latest", params)
        return response.get("data", {})

    def get_5m(self, timestamp: Optional[int] = None) -> Dict:
        """
        Fetch 5-minute averaged price data.

        Args:
            timestamp: Optional Unix timestamp to fetch historical 5m data.
                      Must be divisible by 300.

        Returns:
            Dictionary with item IDs as keys and averaged data:
            {
                "item_id": {
                    "avgHighPrice": int,    # Average instant-buy price
                    "highPriceVolume": int, # Volume of instant-buys
                    "avgLowPrice": int,     # Average instant-sell price
                    "lowPriceVolume": int   # Volume of instant-sells
                }
            }
        """
        params = {"timestamp": timestamp} if timestamp else None
        response = self._request("/5m", params)
        return response.get("data", {})

    def get_1h(self, timestamp: Optional[int] = None) -> Dict:
        """
        Fetch 1-hour averaged price data.

        Args:
            timestamp: Optional Unix timestamp to fetch historical 1h data.
                      Must be divisible by 3600.

        Returns:
            Same format as get_5m() but with hourly averages.
        """
        params = {"timestamp": timestamp} if timestamp else None
        response = self._request("/1h", params)
        return response.get("data", {})

    def get_timeseries(
        self,
        item_id: int,
        timestep: str = "1h"
    ) -> List[Dict]:
        """
        Fetch historical timeseries data for a specific item.

        Args:
            item_id: Item ID to fetch history for
            timestep: One of "5m", "1h", "6h", "24h"

        Returns:
            List of historical data points (up to 365 points):
            [
                {
                    "timestamp": int,
                    "avgHighPrice": int,
                    "avgLowPrice": int,
                    "highPriceVolume": int,
                    "lowPriceVolume": int
                },
                ...
            ]
        """
        if timestep not in self.VALID_TIMESTEPS:
            raise ValueError(f"Invalid timestep '{timestep}'. Must be one of {self.VALID_TIMESTEPS}")

        response = self._request("/timeseries", {"id": item_id, "timestep": timestep})
        return response.get("data", [])

    def get_name_for_id(self, item_id: Union[int, str]) -> Optional[str]:
        """Get item name from ID."""
        return self._id_to_name_map.get(str(item_id))

    def get_id_for_name(self, name: str) -> Optional[str]:
        """Get item ID from name."""
        return self._name_to_id_map.get(name)

    def get_item_metadata(self, item_id: Union[int, str]) -> Optional[Dict]:
        """Get full item metadata including buy limit, alch values, etc."""
        return self._item_metadata.get(str(item_id))

    def set_name_to_id_mapping(
        self,
        name_to_id: Dict[str, str],
        id_to_name: Dict[str, str]
    ):
        """Set name/ID mappings manually (e.g., from cached file)."""
        self._name_to_id_map = name_to_id
        self._id_to_name_map = id_to_name


class HistoricalGrandExchangeClient:
    """
    Client for replaying historical GE data from local files.

    Used for backtesting and training without hitting the live API.
    """

    def __init__(self, data_dir: str = "5m", random_start: bool = False):
        """
        Initialize the historical client.

        Args:
            data_dir: Directory containing historical data files
            random_start: Whether to start at a random position in the data
        """
        import os
        import random

        self._data_dir = data_dir
        self._data_files: List[str] = []
        self._current_index = 0
        self._current_data: Dict = {}

        # ID/name mappings
        self._name_to_id_map: Dict[str, str] = {}
        self._id_to_name_map: Dict[str, str] = {}

        # Load data files
        if os.path.exists(data_dir):
            self._data_files = sorted([
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith('.json')
            ])

        if random_start and self._data_files:
            self._current_index = random.randint(0, len(self._data_files) - 1)

        # Load initial data
        self._load_current_data()

        logger.info(
            f"HistoricalGrandExchangeClient initialized with {len(self._data_files)} files, "
            f"starting at index {self._current_index}"
        )

    def _load_current_data(self):
        """Load data from the current file."""
        import json

        if not self._data_files:
            self._current_data = {}
            return

        try:
            with open(self._data_files[self._current_index], 'r') as f:
                self._current_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading data file: {e}")
            self._current_data = {}

    def get_latest(self) -> Dict:
        """Get current price data."""
        return self._current_data.get("data", self._current_data)

    def advance(self) -> bool:
        """
        Advance to the next data file.

        Returns:
            True if successfully advanced, False if at end (loops back)
        """
        if not self._data_files:
            return False

        self._current_index = (self._current_index + 1) % len(self._data_files)
        self._load_current_data()
        return True

    def reset(self):
        """Reset to the beginning of the data."""
        self._current_index = 0
        self._load_current_data()

    def set_name_to_id_mapping(
        self,
        name_to_id: Dict[str, str],
        id_to_name: Dict[str, str]
    ):
        """Set name/ID mappings."""
        self._name_to_id_map = name_to_id
        self._id_to_name_map = id_to_name

    def get_name_for_id(self, item_id: Union[int, str]) -> Optional[str]:
        """Get item name from ID."""
        return self._id_to_name_map.get(str(item_id))

    def get_id_for_name(self, name: str) -> Optional[str]:
        """Get item ID from name."""
        return self._name_to_id_map.get(name)
