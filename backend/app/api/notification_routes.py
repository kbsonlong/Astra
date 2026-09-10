"""站内提醒 WebSocket API。"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .auth import authorize_websocket

router = APIRouter(prefix="/api/notifications", tags=["notifications"])


async def _notification_snapshot(websocket: WebSocket) -> dict[str, object]:
    notifications = await asyncio.to_thread(
        websocket.app.state.speaker_store.list_notifications
    )
    return {
        "type": "notifications",
        "items": [
            {
                "id": row["notification_id"],
                "type": row["notification_type"],
                "title": "新声纹待审核" if row["notification_type"] == "speaker_review" else "声纹样本待补充确认",
                "message": row["message"],
                "speaker_id": row["speaker_id"],
                "created_at": row["created_at"],
            }
            for row in notifications
        ],
        "unread_count": len(notifications),
    }


@router.websocket("/events")
async def notification_events(websocket: WebSocket) -> None:
    if not await authorize_websocket(websocket):
        return
    await websocket.accept()
    last_payload = ""
    try:
        while True:
            payload = await _notification_snapshot(websocket)
            serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            if serialized != last_payload:
                await websocket.send_json(payload)
                last_payload = serialized
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        return
