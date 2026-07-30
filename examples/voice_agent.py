"""A voice agent that sees the user, both usage modes.

Mode 1 (ambient context): watch() keeps a one-line scene description fresh, and the
agent injects it into the LLM context right before each reply. Cheap, always current,
and the main LLM never waits on vision.

Mode 2 (on demand): a `look` tool the LLM calls when the conversation needs eyes,
including questions about the recent past ("what was I holding 30 seconds ago?").

The two modes are complementary, not redundant: the ambient line gives the agent
standing awareness (who is there, what changed), while the tool answers specific
questions the watch prompt never covers, and can look back in time to verify
something. The instructions below tell the LLM exactly when to reach for the tool,
which prevents it from answering detailed visual questions off the stale ambient
line. If you only need one mode, see examples/tool_calling.py for tool-only.

Run like any LiveKit agent:
    OVERSHOOT_API_KEY=... LIVEKIT_URL=... LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=... \
        python voice_agent.py dev
"""

import os

from livekit import agents
from livekit.agents import Agent, AgentSession, WorkerOptions, cli, llm
from livekit.plugins import openai, silero

from overshoot_livekit import RealtimeVision


class SeeingAgent(Agent):
    def __init__(self, vision_tools: list) -> None:
        super().__init__(
            instructions="You are a friendly assistant that can see the user. "
            "A [current view] line gives you general awareness; use it for "
            "casual references to the scene. For any specific visual question, "
            "anything small or detailed, or anything about the past, call the "
            "look tool instead of guessing from [current view]. "
            "Do not narrate that you have vision.",
            tools=vision_tools,
        )
        self.latest_scene: str | None = None

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        # Mode 1: inject the freshest scene line right before the LLM replies.
        if self.latest_scene:
            turn_ctx.add_message(role="system", content=f"[current view] {self.latest_scene}")


async def entrypoint(ctx: agents.JobContext) -> None:
    await ctx.connect()

    vision = RealtimeVision(
        room=ctx.room,
        participant="user",  # selected by identity, never "first video track"
        api_key=os.environ["OVERSHOOT_API_KEY"],
        instructions="Answer briefly; you are the vision system of a voice assistant.",
    )

    agent = SeeingAgent(
        vision_tools=[
            # Mode 2: on-demand look, including the recent past via seconds_ago.
            vision.as_tool(
                name="look",
                description="Answer a question about what the user's camera shows, "
                "now or in the recent past.",
                window_ms=5000,
                allow_history=True,
            ),
        ]
    )

    def on_scene(result) -> None:
        agent.latest_scene = result.text

    vision.watch(
        prompt="In one short line, what is happening in the picture?",
        interval=2.0,
        on_result=on_scene,
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
