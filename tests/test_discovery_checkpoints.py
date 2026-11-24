import json

from scannerbot.discovery import DiscoveryCheckpointStore, ShrineReplayBackfill


def test_discovery_checkpoints_survive_store_recreation(tmp_path):
    path = tmp_path / "models" / "discovery_checkpoints.json"
    first = DiscoveryCheckpointStore(str(path))
    first.set("shrine_advanced", 123456)
    first.set("shrine_replay", 1790000000)

    second = DiscoveryCheckpointStore(str(path))
    assert second.get("shrine_advanced") == 123456
    assert second.get("shrine_replay") == 1790000000
    second.set("shrine_advanced", 123000)
    assert second.get("shrine_advanced") == 123456


def test_replay_filters_to_launch_events_and_deduplicates_through_handler():
    class Checkpoints:
        def get(self, provider):
            return 0

        def set(self, provider, value):
            pass

    received = []

    async def handler(candidate):
        received.append(candidate)

    replay = ShrineReplayBackfill(handler, Checkpoints())
    event = {
        "action": "create",
        "protocol": "PUMPFUN",
        "mint": "TokenMint111111111111111111111111111111111",
        "timestamp": 1790000000,
        "block": 123,
    }
    import asyncio
    asyncio.run(replay._process_line(json.dumps(event).encode()))
    asyncio.run(replay._process_line(json.dumps({"action": "buy", "mint": event["mint"]}).encode()))
    assert len(received) == 1
    assert received[0].source == "shrine_replay"
    asyncio.run(replay.close())
