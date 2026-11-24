from __future__ import annotations
import logging
import httpx
from typing import Any
log=logging.getLogger(__name__)

class RugCheckClient:
    """Best-effort RugCheck report lookup. Provider errors remain UNKNOWN, never SAFE."""
    def __init__(self, enabled: bool=True):
        self.enabled=enabled
        self.client=httpx.AsyncClient(timeout=12,headers={"User-Agent":"ScannerBot/2.0"})

    async def close(self): await self.client.aclose()

    async def report(self, mint: str) -> dict[str,Any] | None:
        if not self.enabled: return None
        try:
            r=await self.client.get(f"https://api.rugcheck.xyz/v1/tokens/{mint}/report/summary")
            if r.status_code == 404:
                log.debug("RugCheck has no report for mint %s; treating as UNKNOWN", mint)
                return None
            if r.status_code >= 400:
                # Provider errors are never interpreted as a clean security result.
                # Surface the HTTP status so operators can distinguish API failures from token rejections.
                body = " ".join(r.text.split())[:240]
                log.warning("RugCheck HTTP %d for mint %s; security status remains UNKNOWN; response=%s", r.status_code, mint, body or "<empty>")
                return None
            data=r.json()
            if not isinstance(data,dict): return None
            return data
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("RugCheck request failed for mint %s (%s); security status remains UNKNOWN", mint, type(exc).__name__)
            return None

    @staticmethod
    def assess(report: dict[str,Any] | None, max_score: float) -> tuple[bool,str,float|None,list[str]]:
        if report is None: return False,"UNKNOWN",None,["RugCheck report unavailable; security gate not passed"]
        # RugCheck's raw `score` is an unbounded sum; `score_normalised` is the 0-100 risk scale.
        # Never compare the raw sum to the normalized threshold.
        score=report.get("score_normalised")
        try: score=float(score) if score is not None else None
        except (TypeError,ValueError): score=None
        risks=report.get("risks") or []
        reasons=[]
        if report.get("rugged") is True or report.get("isRugged") is True:
            return False,"REJECT",score,["RugCheck marks this token as rugged"]
        for risk in risks:
            if isinstance(risk,dict):
                level=str(risk.get("level") or risk.get("score") or "").lower()
                name=str(risk.get("name") or risk.get("description") or "Risk")
                reasons.append(f"{name} ({level})")
        # RugCheck score is a risk score; larger is worse. A missing score is not a clean bill.
        if score is None: return False,"UNKNOWN",None,reasons+["Risk score missing"]
        critical=any(any(k in x.lower() for k in ("danger","critical","high")) for x in reasons)
        passed=score<=max_score and not critical
        return passed,("PASS" if passed else "REJECT"),score,reasons[:12]
