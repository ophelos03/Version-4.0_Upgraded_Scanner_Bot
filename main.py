import asyncio, logging
from scannerbot.config import settings
from scannerbot.engine import ScannerEngine

async def main():
    settings.validate()
    logging.basicConfig(level=getattr(logging,settings.log_level,logging.INFO),format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    # HTTP request URLs can contain Telegram bot credentials; never emit them at INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    engine=ScannerEngine(settings)
    try: await engine.run()
    finally: await engine.close()

if __name__=="__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: print("ScannerBot stopped.")
