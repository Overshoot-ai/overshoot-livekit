# overshoot-livekit

Real-time vision for [LiveKit Agents](https://docs.livekit.io/agents/), powered by
[Overshoot](https://overshoot.ai). Your voice agent gets eyes in ~10 lines: it can watch the
user's camera continuously, answer visual questions on demand, and look back in time.

```
pip install "overshoot-livekit @ git+https://github.com/Overshoot-ai/overshoot-livekit.git"
```

## Quickstart

```python
import os
from livekit import agents
from livekit.agents import Agent, AgentSession
from overshoot_livekit import RealtimeVision, VisionUnavailable


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    vision = RealtimeVision(
        room=ctx.room,
        participant="user",              # selected by identity, never "first video track"
        api_key=os.environ["OVERSHOOT_API_KEY"],
    )

    assistant = Agent(
        instructions="You are a helpful assistant that can see the user.",
        tools=[
            vision.as_tool(
                name="look",
                description="Answer a visual question about the user",
                window_ms=10_000,
                allow_history=True,      # lets the LLM ask about the past on its own
            ),
        ],
    )

    session = AgentSession(llm=..., stt=..., tts=...)

    # Periodic scene context (1 request/second), never crashes the agent:
    vision.watch(
        prompt="In one line, describe what is happening in the picture.",
        interval=1.0,
        on_result=lambda r: session.history.add_message(role="system", content=r.text),
    )

    await session.start(agent=assistant, room=ctx.room)
```

Everything else (stream creation, keepalive, reconnects, teardown) is handled internally.
When the room disconnects, the Overshoot stream is deleted automatically.

## Asking questions

```python
await vision.ask("Is the user holding anything?")                       # latest frame
await vision.ask("What happened?", window_ms=10_000)                    # last 10 seconds
await vision.ask("What happened?", start_ms=t0, end_ms=t1)              # absolute window (epoch ms)
await vision.ask("Summarize.", window_ms=60_000, sample_fps=2)          # control frame sampling
```

History reaches back as far as the stream's retention window (10 minutes by default). Asking
for an earlier range returns whatever frames still exist rather than erroring.

## Structured output

Pass a JSON-schema dict (returns a `dict`) or a pydantic model class (returns an instance):

```python
from pydantic import BaseModel

class Scene(BaseModel):
    person_present: bool
    description: str

scene = await vision.ask("Describe the scene.", schema=Scene)

vision.watch(
    prompt="Report the scene.",
    interval=2.0,
    schema=Scene,
    on_result=lambda r: print(r.data.person_present),
)
```

## Failure behavior

Vision failure never crashes the agent:

- `ask()` and tool calls raise `VisionUnavailable` (subclass `VisionSchemaError` for schema
  parse failures) so the agent can say "I cannot see you right now."
- `watch()` skips the tick and retries on rate limits, server errors, and timeouts; it stops
  with one log line on non-retryable errors. It never overlaps requests: a slow request skips
  ticks instead of queueing.

## Configuration

```python
RealtimeVision(
    room, participant, api_key,
    model="...",                # default model for all requests
    instructions="...",         # system prompt prepended to every request
    track_source="camera",      # or "screen_share"
    max_fps=10,                 # frame forwarding cap
    max_output_tokens=256,      # default response cap
)
```

`ask()` accepts per-call overrides for `model`, `max_output_tokens`, and `timeout`, plus an
`extra` dict merged into the request body for anything else the OpenAI-compatible endpoint
accepts (for example `{"temperature": 0}`).
