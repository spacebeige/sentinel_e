"""Asterisk ARI WebSocket client for AWAAZ."""

import asyncio
import json
import os
import aiohttp
import websockets
from typing import Dict, Any, Optional, Callable, List
from urllib.parse import quote
import logging

logger = logging.getLogger(__name__)


class ARIClient:
    """Simplified ARI client for AWAAZ - connects to Asterisk via WebSocket."""

    def __init__(
        self,
        username: str,
        password: str,
        host: str,
        ari_port: int,
        stasis_app: str,
        ssl_verify: bool = True,
    ):
        self.username = username
        self.password = password
        self.host = host
        self.ari_port = ari_port
        self.stasis_app = stasis_app
        self.ssl_verify = ssl_verify

        self.http_base = f"http://{host}:{ari_port}/ari"
        safe_user = quote(username)
        safe_pass = quote(password)
        self.ws_url = (
            f"ws://{host}:{ari_port}/ari/events"
            f"?api_key={safe_user}:{safe_pass}&app={stasis_app}&subscribeAll=true"
        )

        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.running = False
        self.event_handlers: Dict[str, List[Callable]] = {}

    async def connect(self) -> bool:
        """Connect to Asterisk ARI WebSocket."""
        try:
            logger.info(f"Connecting to ARI at {self.host}:{self.ari_port}")
            
            # Test HTTP connection first
            if not self.http_session:
                self.http_session = aiohttp.ClientSession(
                    auth=aiohttp.BasicAuth(self.username, self.password)
                )

            async with self.http_session.get(f"{self.http_base}/asterisk/info") as resp:
                if resp.status != 200:
                    logger.error(f"ARI HTTP endpoint failed: {resp.status}")
                    return False
                logger.info("ARI HTTP endpoint OK")

            # Connect WebSocket
            self.websocket = await websockets.connect(self.ws_url)
            self.running = True
            logger.info("ARI WebSocket connected")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to ARI: {e}")
            if self.http_session:
                await self.http_session.close()
                self.http_session = None
            return False

    async def disconnect(self):
        """Disconnect from ARI."""
        self.running = False
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        if self.http_session:
            await self.http_session.close()
            self.http_session = None
        logger.info("ARI disconnected")

    async def listen_events(self, handler: Callable[[str, Dict], None]):
        """Listen for ARI events and dispatch to handler."""
        if not self.websocket:
            logger.error("WebSocket not connected")
            return

        try:
            async for message in self.websocket:
                try:
                    event = json.loads(message)
                    event_type = event.get("type", "Unknown")
                    await handler(event_type, event)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse ARI event: {message}")
        except Exception as e:
            logger.error(f"ARI listener error: {e}")
            self.running = False

    async def send_command(self, method: str, resource: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Send HTTP command to ARI."""
        if not self.http_session:
            return {"status": 500, "error": "HTTP session not connected"}

        url = f"{self.http_base}/{resource}"
        try:
            async with self.http_session.request(method, url, json=data) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    logger.error(f"ARI error {resp.status}: {text}")
                    return {"status": resp.status, "error": text}
                if resp.status == 204:
                    return {"status": 204}
                return await resp.json()
        except Exception as e:
            logger.error(f"ARI HTTP request failed: {e}")
            return {"status": 500, "error": str(e)}

    async def answer_channel(self, channel_id: str) -> bool:
        """Answer a channel."""
        logger.info(f"Answering channel {channel_id}")
        resp = await self.send_command("POST", f"channels/{channel_id}/answer")
        return resp.get("status", 500) < 400

    async def hangup_channel(self, channel_id: str) -> bool:
        """Hang up a channel."""
        logger.info(f"Hanging up channel {channel_id}")
        resp = await self.send_command("DELETE", f"channels/{channel_id}")
        status = resp.get("status", 500)
        return status < 400 or status == 404  # 404 = already hung up

    async def get_caller_ani(self, channel_id: str) -> str:
        """Get caller phone number from channel variable."""
        try:
            resp = await self.send_command("GET", f"channels/{channel_id}/variable?variable=CALLERID(num)")
            ani = resp.get("value", "")
            return ani if isinstance(ani, str) else str(ani)
        except Exception as e:
            logger.warning(f"Failed to get ANI for {channel_id}: {e}")
            return ""

    async def play_sound(self, channel_id: str, sound_file: str) -> Optional[str]:
        """Play sound file on channel. Returns playback_id if successful."""
        if not sound_file.startswith("sound:"):
            sound_file = f"sound:{sound_file}"
        
        logger.info(f"Playing {sound_file} on {channel_id}")
        resp = await self.send_command("POST", f"channels/{channel_id}/play", data={"media": sound_file})
        
        if resp.get("id"):
            return resp["id"]
        return None

    async def set_channel_variable(self, channel_id: str, var_name: str, var_value: str) -> bool:
        """Set a channel variable."""
        resp = await self.send_command(
            "POST",
            f"channels/{channel_id}/variable",
            data={"variable": var_name, "value": var_value},
        )
        return resp.get("status", 500) < 400
