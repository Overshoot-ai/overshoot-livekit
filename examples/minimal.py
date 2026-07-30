"""Smallest possible integration: no agents framework, just a room and questions.

Joins a LiveKit room, watches the participant "user", prints a scene line every
2 seconds, then asks one question about the last 10 seconds and one structured query.

    OVERSHOOT_API_KEY=... LIVEKIT_URL=... LIVEKIT_TOKEN=... python minimal.py
"""

import asyncio
import os

from livekit import rtc
from pydantic import BaseModel

from overshoot_livekit import RealtimeVision, VisionUnavailable


class Scene(BaseModel):
    person_present: bool
    description: str


async def main() -> None:
    room = rtc.Room()
    await room.connect(os.environ["LIVEKIT_URL"], os.environ["LIVEKIT_TOKEN"])

    vision = RealtimeVision(
        room=room,
        participant="user",
        api_key=os.environ["OVERSHOOT_API_KEY"],
    )
    vision.watch(
        prompt="In one line, what is happening?",
        interval=2.0,
        on_result=lambda r: print(f"[watch] {r.text}"),
    )

    await asyncio.sleep(30)

    try:
        print("[ask]", await vision.ask("What happened?", window_ms=10_000))
        scene = await vision.ask("Describe the scene.", schema=Scene)
        print("[structured]", scene.person_present, "-", scene.description)
    except VisionUnavailable as exc:
        print("vision unavailable:", exc)

    await vision.aclose()
    await room.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
