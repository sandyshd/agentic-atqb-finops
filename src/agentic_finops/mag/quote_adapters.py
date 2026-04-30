from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import quote
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


@dataclass
class ProviderQuote:
    provider: str
    spot_multiplier: float
    unit_price_per_gpu_hour: float
    egress_per_1k_tokens: float
    latency_ms: float
    source: str
    as_of: str


class AzureQuoteAdapter:
    """Fetch Azure VM spot pricing from the public Retail Prices API."""

    def __init__(self) -> None:
        self.region = os.getenv("BROKER_AZURE_REGION", "eastus")
        self.reference_price = max(0.0001, float(os.getenv("BROKER_REFERENCE_GPU_HOURLY_USD", "1.0")))
        self.egress_per_1k_tokens = max(0.0, float(os.getenv("BROKER_AZURE_EGRESS_PER_1K_TOKENS", "0.0001")))
        self.latency_ms = max(1.0, float(os.getenv("BROKER_AZURE_LATENCY_MS", "40")))

    def collect(self) -> ProviderQuote | None:
        spot_filt = (
            "serviceName eq 'Virtual Machines'"
            + f" and armRegionName eq '{self.region}'"
            + " and contains(skuName,'Spot')"
            + " and priceType eq 'Consumption'"
        )
        url = "https://prices.azure.com/api/retail/prices?$filter=" + quote(spot_filt, safe="=$'()")

        min_price = None
        next_url = url
        pages = 0
        while next_url and pages < 3:
            pages += 1
            try:
                req = Request(next_url, headers={"User-Agent": "agentic-finops/1.0"})
                with urlopen(req, timeout=20) as resp:
                    payload = json.loads(resp.read())
            except Exception as exc:
                logger.warning("AzureQuoteAdapter: retail API fetch failed: %s", exc)
                return None

            for item in payload.get("Items", []):
                sku = str(item.get("skuName", "")).lower()
                meter = str(item.get("meterName", "")).lower()
                if "spot" not in sku and "spot" not in meter and "low priority" not in sku:
                    continue
                price = float(item.get("retailPrice", 0.0) or 0.0)
                if price <= 0:
                    continue
                min_price = price if min_price is None else min(min_price, price)

            next_url = payload.get("NextPageLink")

        source = "azure-retail-prices-spot"
        if min_price is None:
            # Fallback: use the lowest VM retail price to keep broker model live.
            any_vm_filt = (
                "serviceName eq 'Virtual Machines'"
                + f" and armRegionName eq '{self.region}'"
                + " and priceType eq 'Consumption'"
            )
            fallback_url = "https://prices.azure.com/api/retail/prices?$filter=" + quote(any_vm_filt, safe="=$'()")
            try:
                req = Request(fallback_url, headers={"User-Agent": "agentic-finops/1.0"})
                with urlopen(req, timeout=20) as resp:
                    payload = json.loads(resp.read())
                prices = [float(i.get("retailPrice", 0.0) or 0.0) for i in payload.get("Items", [])]
                prices = [p for p in prices if p > 0]
                if prices:
                    min_price = min(prices)
                    source = "azure-retail-prices-fallback"
            except Exception as exc:
                logger.warning("AzureQuoteAdapter: retail fallback fetch failed: %s", exc)
                return None

        if min_price is None:
            return None

        return ProviderQuote(
            provider="azure",
            unit_price_per_gpu_hour=min_price,
            spot_multiplier=max(0.1, min(2.5, min_price / self.reference_price)),
            egress_per_1k_tokens=self.egress_per_1k_tokens,
            latency_ms=self.latency_ms,
            source=source,
            as_of=datetime.now(timezone.utc).isoformat(),
        )


class AwsQuoteAdapter:
    """Fetch AWS EC2 spot pricing using boto3 credentials when available."""

    def __init__(self) -> None:
        self.region = os.getenv("BROKER_AWS_REGION", "us-east-1")
        self.instance_type = os.getenv("BROKER_AWS_INSTANCE_TYPE", "g5.xlarge")
        self.reference_price = max(0.0001, float(os.getenv("BROKER_REFERENCE_GPU_HOURLY_USD", "1.0")))
        self.egress_per_1k_tokens = max(0.0, float(os.getenv("BROKER_AWS_EGRESS_PER_1K_TOKENS", "0.0004")))
        self.latency_ms = max(1.0, float(os.getenv("BROKER_AWS_LATENCY_MS", "55")))

    def collect(self) -> ProviderQuote | None:
        try:
            import boto3  # type: ignore
        except Exception:
            return None

        try:
            session = boto3.session.Session(region_name=self.region)
            ec2 = session.client("ec2", region_name=self.region)
            resp = ec2.describe_spot_price_history(
                InstanceTypes=[self.instance_type],
                ProductDescriptions=["Linux/UNIX"],
                MaxResults=20,
            )
            prices = [float(r["SpotPrice"]) for r in resp.get("SpotPriceHistory", []) if "SpotPrice" in r]
            if not prices:
                return None
            spot = min(prices)
        except Exception as exc:
            logger.warning("AwsQuoteAdapter: spot API fetch failed: %s", exc)
            return None

        return ProviderQuote(
            provider="aws",
            unit_price_per_gpu_hour=spot,
            spot_multiplier=max(0.1, min(2.5, spot / self.reference_price)),
            egress_per_1k_tokens=self.egress_per_1k_tokens,
            latency_ms=self.latency_ms,
            source="aws-ec2-spot-price-history",
            as_of=datetime.now(timezone.utc).isoformat(),
        )


class NeoCloudQuoteAdapter:
    """Fetch neocloud quote from a configured endpoint returning JSON."""

    def __init__(self) -> None:
        self.url = os.getenv("BROKER_NEOCLOUD_QUOTE_URL", "").strip()
        self.reference_price = max(0.0001, float(os.getenv("BROKER_REFERENCE_GPU_HOURLY_USD", "1.0")))
        self.default_egress = max(0.0, float(os.getenv("BROKER_NEO_EGRESS_PER_1K_TOKENS", "0.0012")))
        self.default_latency = max(1.0, float(os.getenv("BROKER_NEO_LATENCY_MS", "70")))

    def collect(self) -> ProviderQuote | None:
        if not self.url:
            return None
        try:
            req = Request(self.url, headers={"User-Agent": "agentic-finops/1.0"})
            with urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read())
        except Exception as exc:
            logger.warning("NeoCloudQuoteAdapter: quote endpoint fetch failed: %s", exc)
            return None

        price = float(payload.get("price_per_gpu_hour", 0.0) or 0.0)
        mult = float(payload.get("spot_multiplier", 0.0) or 0.0)
        if mult <= 0 and price > 0:
            mult = price / self.reference_price
        if mult <= 0:
            return None

        return ProviderQuote(
            provider="neocloud-gpu",
            unit_price_per_gpu_hour=price if price > 0 else (self.reference_price * mult),
            spot_multiplier=max(0.1, min(2.5, mult)),
            egress_per_1k_tokens=float(payload.get("egress_per_1k_tokens", self.default_egress) or self.default_egress),
            latency_ms=float(payload.get("latency_ms", self.default_latency) or self.default_latency),
            source=str(payload.get("source", "neocloud-quote-endpoint")),
            as_of=datetime.now(timezone.utc).isoformat(),
        )


class ProviderQuoteManager:
    def __init__(self) -> None:
        self.adapters = [
            AzureQuoteAdapter(),
            AwsQuoteAdapter(),
            NeoCloudQuoteAdapter(),
        ]

    def collect(self) -> list[ProviderQuote]:
        quotes: list[ProviderQuote] = []
        for adapter in self.adapters:
            try:
                quote = adapter.collect()
                if quote is not None:
                    quotes.append(quote)
            except Exception as exc:
                logger.warning("ProviderQuoteManager: adapter failure: %s", exc)
        return quotes
