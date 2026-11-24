from dataclasses import dataclass, field
from typing import Any

@dataclass
class Candidate:
    chain_id: str
    token_address: str
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    pair_address: str = ""
    created_at: float | None = None

@dataclass
class TokenAnalysis:
    chain_id: str
    token_address: str
    symbol: str
    name: str
    pair_address: str
    dex_id: str
    url: str
    price_usd: float
    liquidity_usd: float
    volume_h1: float
    volume_h24: float
    buys_h1: int
    sells_h1: int
    buys_h24: int
    sells_h24: int
    price_change_h1: float
    market_cap: float | None
    fdv: float | None
    age_hours: float | None
    score: float
    decision: str
    reasons: list[str]
    risk_flags: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    dexpaprika_url: str = ""
    market_cap_source: str = "market_cap"


_SOLANA_BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def solana_mint_validation_reason(value: str) -> str | None:
    """Return None for a valid mint, otherwise a non-sensitive reason code."""
    if not isinstance(value, str) or not value:
        return "empty_or_non_string"
    if not 32 <= len(value) <= 44:
        return "invalid_character_count"
    number = 0
    for char in value:
        digit = _SOLANA_BASE58_ALPHABET.find(char)
        if digit < 0:
            return "invalid_base58_character"
        number = number * 58 + digit
    decoded = number.to_bytes((number.bit_length() + 7) // 8, "big") if number else b""
    leading_zeroes = len(value) - len(value.lstrip("1"))
    decoded = (bytes([0]) * leading_zeroes) + decoded
    if len(decoded) != 32:
        return "decoded_length_not_32_bytes"
    return None


def is_valid_solana_mint(value: str) -> bool:
    """Return True only for canonical base58 addresses decoding to 32 bytes."""
    return solana_mint_validation_reason(value) is None


def canonical_token_key(chain_id: str, address: str) -> str:
    """Solana mints are case-sensitive base58: keep them verbatim.
    Other (hex) chains are case-insensitive, so they keep lowercase keys."""
    return address if chain_id == "solana" else address.lower()
