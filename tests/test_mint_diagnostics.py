from scannerbot.discovery import ShrineDiscovery
from scannerbot.models import is_valid_solana_mint, solana_mint_validation_reason


def test_solana_mint_validation_reason_classifies_invalid_values():
    assert solana_mint_validation_reason("") == "empty_or_non_string"
    assert solana_mint_validation_reason("short") == "invalid_character_count"
    assert solana_mint_validation_reason("0" * 32) == "invalid_base58_character"
    assert solana_mint_validation_reason("z" * 44) == "decoded_length_not_32_bytes"
    assert solana_mint_validation_reason("So11111111111111111111111111111111111111112") is None
    assert is_valid_solana_mint("So11111111111111111111111111111111111111112")


def test_shrine_candidate_records_which_field_supplied_mint():
    candidate = ShrineDiscovery._candidate(
        {"address": "So11111111111111111111111111111111111111112"},
        "shrine",
    )
    assert candidate.token_address == "So11111111111111111111111111111111111111112"
    assert candidate.metadata["_scanner_mint_field"] == "address"


def test_shrine_candidate_marks_missing_mint_field():
    candidate = ShrineDiscovery._candidate({"pool": "pool-address"}, "shrine")
    assert candidate.token_address == ""
    assert candidate.metadata["_scanner_mint_field"] == "missing"
