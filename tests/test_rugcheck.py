from scannerbot.security.rugcheck import RugCheckClient

def test_rugcheck_unknown_fails_closed():
    passed,status,score,reasons=RugCheckClient.assess(None,50)
    assert not passed and status=="UNKNOWN" and score is None

def test_rugcheck_high_score_rejected():
    passed,status,score,reasons=RugCheckClient.assess({"score":9000,"score_normalised":90,"risks":[]},50)
    assert not passed and status=="REJECT"


def test_rugcheck_uses_normalised_score_not_raw_sum():
    passed,status,score,reasons=RugCheckClient.assess({"score":1272,"score_normalised":24,"risks":[]},50)
    assert passed and status=="PASS" and score==24

def test_rugged_flag_rejected_even_if_score_low():
    passed,status,score,reasons=RugCheckClient.assess({"score":10,"score_normalised":1,"rugged":True,"risks":[]},50)
    assert not passed and status=="REJECT"


from scannerbot.models import is_valid_solana_mint


def test_valid_solana_public_keys_are_accepted():
    assert is_valid_solana_mint("11111111111111111111111111111111")
    assert is_valid_solana_mint("So11111111111111111111111111111111111111112")


def test_malformed_solana_mints_are_rejected():
    assert not is_valid_solana_mint("")
    assert not is_valid_solana_mint("0" * 32)  # zero is not in the Base58 alphabet
    assert not is_valid_solana_mint("1111111111111111111111111111111")  # 31 bytes
    assert not is_valid_solana_mint("111111111111111111111111111111111")  # 33 bytes
