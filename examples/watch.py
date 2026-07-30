"""Ambient vision: watch() keeps a scene line fresh, injected before each LLM turn.

The agent always knows roughly what the camera shows, and the main LLM never waits
on vision. This is the recommended starting point for conversational agents.

Run like any LiveKit agent (or `python watch.py console` for local testing):
    OVERSHOOT_API_KEY=... LIVEKIT_URL=... LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=... \
        python watch.py dev
"""

import os

from livekit import agents
from livekit.agents import Agent, AgentSession, WorkerOptions, cli, llm
from livekit.plugins import openai, silero

from overshoot_livekit import RealtimeVision


class SeeingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a friendly assistant that can see the user. "
            "A [current view] line tells you what the camera currently shows; "
            "use it naturally and do not narrate that you have vision."
        )
        self.latest_scene: str | None = None

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        # Inject the freshest scene line right before the LLM replies.
        if self.latest_scene:
            turn_ctx.add_message(role="system", content=f"[current view] {self.latest_scene}")


async def entrypoint(ctx: agents.JobContext) -> None:
    await ctx.connect()

    vision = RealtimeVision(
        room=ctx.room,
        participant="user",  # selected by identity, never "first video track"
        api_key=os.environ["OVERSHOOT_API_KEY"],
    )

    agent = SeeingAgent()

    vision.watch(
        prompt="In one short line, what is happening in the picture?",
        interval=2.0,
        on_result=lambda r: setattr(agent, "latest_scene", r.text),
    )

    session = AgentSession(
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4.1-mini"),
        tts=openai.TTS(),
        vad=silero.VAD.load(),
    )
    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
