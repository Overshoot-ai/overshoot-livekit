"""One-off questions: no agents framework, just a room and ask().

Joins a LiveKit room, waits for the participant "user" to publish, then asks about
the latest frame, a trailing window, and once with structured output.

    OVERSHOOT_API_KEY=... LIVEKIT_URL=... LIVEKIT_TOKEN=... python ask.py
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

    await asyncio.sleep(15)  # let some video accumulate

    try:
        print("[latest]", await vision.ask("What do you see? One line."))
        print("[window]", await vision.ask("What happened?", window_ms=10_000))
        scene = await vision.ask("Describe the scene.", schema=Scene)
        print("[structured]", scene.person_present, "-", scene.description)
    except VisionUnavailable as exc:
        print("vision unavailable:", exc)

    await vision.aclose()
    await room.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
