import asyncio
import logging

import httpx
import pytest

from scannerbot.telegram import Telegram


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.is_error = status_code >= 400
        self._payload = payload or {"ok": True, "result": {"message_id": 42}}

    def json(self):
        return self._payload

    def raise_for_status(self):
        request = httpx.Request("POST", "https://api.telegram.org/botsecret/sendMessage")
        response = httpx.Response(self.status_code, request=request)
        raise httpx.HTTPStatusError("HTTP error", request=request, response=response)


class FakeClient:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    async def post(self, *args, **kwargs):
        if self.error:
            raise self.error
        return self.result


def make_bot():
    bot = Telegram("secret-token", "123")
    bot.client = FakeClient(result=FakeResponse())
    return bot


def test_send_logs_success(caplog):
    bot = make_bot()
    with caplog.at_level(logging.INFO):
        asyncio.run(bot.send("hello"))
    assert "Telegram send succeeded" in caplog.text
    assert "message_id=42" in caplog.text
    assert "secret-token" not in caplog.text


def test_send_logs_timeout_and_redacts_token(caplog):
    bot = make_bot()
    request = httpx.Request("POST", "https://api.telegram.org/botsecret-token/sendMessage")
    bot.client = FakeClient(error=httpx.ReadTimeout("timed out", request=request))
    with caplog.at_level(logging.WARNING):
        with pytest.raises(httpx.ReadTimeout):
            asyncio.run(bot.send("hello"))
    assert "category=timeout" in caplog.text
    assert "secret-token" not in caplog.text


def test_send_logs_telegram_rejection(caplog):
    bot = make_bot()
    bot.client = FakeClient(result=FakeResponse(payload={
        "ok": False, "error_code": 400, "description": "chat not found"
    }))
    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match="error_code=400"):
            asyncio.run(bot.send("hello"))
    assert "category=telegram_rejection" in caplog.text
    assert "chat not found" in caplog.text
