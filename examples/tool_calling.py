"""Tool calling only: the LLM decides when to look.

No ambient watch loop here. The agent has one extra tool, `look`, and calls it
whenever the conversation needs eyes. With allow_history=True the tool schema
includes an optional seconds_ago parameter, so questions like "what was I holding
30 seconds ago?" work without any extra code: the LLM fills the parameter itself.

Run like any LiveKit agent (or `python tool_calling.py console` for local testing):
    OVERSHOOT_API_KEY=... LIVEKIT_URL=... LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=... \
        python tool_calling.py dev
"""

import os

from livekit import agents
from livekit.agents import Agent, AgentSession, WorkerOptions, cli
from livekit.plugins import openai, silero

from overshoot_livekit import RealtimeVision


async def entrypoint(ctx: agents.JobContext) -> None:
    await ctx.connect()

    vision = RealtimeVision(
        room=ctx.room,
        participant="user",
        api_key=os.environ["OVERSHOOT_API_KEY"],
    )

    agent = Agent(
        instructions="You are a helpful voice assistant. When a question depends on "
        "what the user's camera shows, now or recently, use the look tool.",
        tools=[
            vision.as_tool(
                name="look",
                description="Answer a question about what the user's camera shows. "
                "Pass seconds_ago to ask about the recent past.",
                window_ms=5000,
                allow_history=True,
            ),
        ],
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
