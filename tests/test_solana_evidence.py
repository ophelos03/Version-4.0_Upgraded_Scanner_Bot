from scannerbot.early_momentum import SolanaEvidenceClient


def test_public_rpc_uses_multiple_fallback_endpoints():
    client = SolanaEvidenceClient()
    assert client.rpc_urls[0] == "https://solana-rpc.publicnode.com"
    assert "https://public.rpc.solanavibestation.com" in client.rpc_urls
    assert "https://solana.leorpc.com/?api_key=FREE" in client.rpc_urls
    assert "https://public-solana-mainnet.rpcnode.dev" in client.rpc_urls
    assert "https://rpc.nodeflare.app/solana/public" in client.rpc_urls
    assert "https://api.mainnet-beta.solana.com" in client.rpc_urls
    assert len(client.rpc_urls) >= 5


def test_helius_is_optional_and_falls_back_to_public_rpc():
    client = SolanaEvidenceClient(helius_api_key="test-key")
    assert client.rpc_urls[0].startswith("https://mainnet.helius-rpc.com/")
    assert "https://public.rpc.solanavibestation.com" in client.rpc_urls
