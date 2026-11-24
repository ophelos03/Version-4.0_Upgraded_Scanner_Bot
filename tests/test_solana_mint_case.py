from scannerbot.database import Database
from scannerbot.models import Candidate, canonical_token_key, is_valid_solana_mint

WSOL = "So11111111111111111111111111111111111111112"
MIXED = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


def test_solana_mint_case_is_preserved_in_candidates(tmp_path):
    db = Database(str(tmp_path / "case.db"))
    db.register_candidate(Candidate("solana", WSOL, "test"))
    rows = db.active_candidates(60, 10)
    assert [r["token_address"] for r in rows] == [WSOL]
    db.close()


def test_mints_differing_only_by_case_are_distinct(tmp_path):
    db = Database(str(tmp_path / "distinct.db"))
    db.register_candidate(Candidate("solana", MIXED, "test"))
    db.register_candidate(Candidate("solana", MIXED.lower(), "test"))
    stored = {r["token_address"] for r in db.active_candidates(60, 10)}
    assert stored == {MIXED, MIXED.lower()}
    db.close()


def test_is_valid_solana_mint():
    assert is_valid_solana_mint(WSOL)
    assert is_valid_solana_mint(MIXED)
    # 44 chars of base58 that decode to 33 bytes (the "invalid length" case)
    assert not is_valid_solana_mint("z" * 44)


def test_canonical_token_key():
    assert canonical_token_key("solana", "AbC") == "AbC"
    assert canonical_token_key("ethereum", "AbC") == "abc"
