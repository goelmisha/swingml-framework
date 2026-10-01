"""Data acquisition layer: universe, prices, delivery, caching.

Providers are pluggable so the pipeline can run fully offline (mock) or
against real NSE sources without any downstream code change.
"""

from swingml.data.cache import DiskCache
from swingml.data.delivery import (
    DeliveryProvider,
    MockDeliveryProvider,
    NseBhavcopyDeliveryProvider,
    make_delivery_provider,
)
from swingml.data.prices import (
    MockPriceProvider,
    PriceProvider,
    YFinancePriceProvider,
    make_price_provider,
)

__all__ = [
    "DiskCache",
    "DeliveryProvider",
    "MockDeliveryProvider",
    "NseBhavcopyDeliveryProvider",
    "make_delivery_provider",
    "MockPriceProvider",
    "PriceProvider",
    "YFinancePriceProvider",
    "make_price_provider",
]
