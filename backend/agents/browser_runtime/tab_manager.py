from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger("BrowserRuntime.Tabs")


@dataclass
class BrowserSession:
    session_id: str
    browser: Any
    context: Any
    page: Any
    created_at: float
    last_used_at: float


class BrowserTabManager:
    """Owns lightweight Playwright sessions for the embedded runtime."""

    def __init__(self, headless: bool = True, ttl_seconds: int = 900):
        self.headless = headless
        self.ttl_seconds = ttl_seconds
        self._playwright = None
        self._sessions: Dict[str, BrowserSession] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, session_id: Optional[str] = None, *, cdp_url: Optional[str] = None) -> BrowserSession:
        async with self._lock:
            self._evict_expired_locked()
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.last_used_at = time.monotonic()
                return session

            playwright = await self._get_playwright()
            if cdp_url:
                browser = await playwright.chromium.connect_over_cdp(cdp_url)
            else:
                browser = await playwright.chromium.launch(headless=self.headless)
            context = await browser.new_context(
                viewport={"width": 1280, "height": 900},
                ignore_https_errors=True,
            )
            page = await context.new_page()
            resolved_session_id = session_id or str(uuid.uuid4())
            session = BrowserSession(
                session_id=resolved_session_id,
                browser=browser,
                context=context,
                page=page,
                created_at=time.monotonic(),
                last_used_at=time.monotonic(),
            )
            self._sessions[resolved_session_id] = session
            return session

    async def new_tab(self, session_id: str, url: Optional[str] = None) -> BrowserSession:
        session = await self.get_or_create(session_id)
        session.page = await session.context.new_page()
        if url:
            await session.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        session.last_used_at = time.monotonic()
        return session

    async def close(self, session_id: str) -> bool:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if not session:
            return False
        try:
            await session.context.close()
            await session.browser.close()
        except Exception as exc:
            logger.debug("Browser session close failed: %s", exc)
        return True

    async def close_all(self) -> None:
        for session_id in list(self._sessions.keys()):
            await self.close(session_id)
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    def list_sessions(self) -> Dict[str, Dict[str, Any]]:
        now = time.monotonic()
        return {
            session_id: {
                "session_id": session_id,
                "age_seconds": round(now - session.created_at, 1),
                "idle_seconds": round(now - session.last_used_at, 1),
                "url": getattr(session.page, "url", ""),
            }
            for session_id, session in self._sessions.items()
        }

    async def _get_playwright(self):
        if self._playwright is None:
            try:
                from playwright.async_api import async_playwright
            except Exception as exc:
                raise RuntimeError(
                    "Playwright is not installed. Install backend requirements and run `playwright install chromium`."
                ) from exc
            self._playwright = await async_playwright().start()
        return self._playwright

    def _evict_expired_locked(self) -> None:
        now = time.monotonic()
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.last_used_at > self.ttl_seconds
        ]
        for session_id in expired:
            session = self._sessions.pop(session_id, None)
            if session:
                asyncio.create_task(self._close_session_objects(session))

    @staticmethod
    async def _close_session_objects(session: BrowserSession) -> None:
        try:
            await session.context.close()
            await session.browser.close()
        except Exception:
            pass


_tab_manager: Optional[BrowserTabManager] = None


def get_tab_manager() -> BrowserTabManager:
    global _tab_manager
    if _tab_manager is None:
        _tab_manager = BrowserTabManager()
    return _tab_manager
