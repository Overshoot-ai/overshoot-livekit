"""RealtimeVision: give a LiveKit agent eyes via the Overshoot v1beta API.

One class, three affordances:
- ask()      one visual question (latest frame, trailing window, or absolute window)
- as_tool()  ask() wrapped as a LiveKit Agents function tool
- watch()    sequential background loop feeding results to a callback

v0 ingest is the republish bridge: subscribe to the target participant's track in the
customer room, throttle, and republish frames into the Overshoot stream's LiveKit room.
When the backend `livekit_room` source ships, the bridge collapses into one create-stream
call without changing this API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Callable

import httpx
from livekit import rtc

from .errors import VisionSchemaError, VisionUnavailable

log = logging.getLogger("overshoot_livekit")

DEFAULT_BASE_URL = "https://api.overshoot.ai/v1beta"
DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"

_TRACK_SOURCES = {
    "camera": rtc.TrackSource.SOURCE_CAMERA,
    "screen_share": rtc.TrackSource.SOURCE_SCREENSHARE,
}
# 4xx that will not succeed on retry: watch() stops instead of hammering.
_FATAL_STATUSES = {400, 401, 403, 404, 422}


@dataclass(frozen=True)
class VisionResult:
    """One watch() result: text is always set; data only when a schema was given."""

    text: str
    data: Any = None


class _Api:
    """Minimal async client for the Overshoot v1beta HTTP surface."""

    def __init__(self, api_key: str, base_url: str) -> None:
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15.0,
        )

    async def _request(self, method: str, path: str, *, json_body: dict | None = None,
                       timeout: float | None = None) -> dict:
        try:
            resp = await self._http.request(method, path, json=json_body, timeout=timeout)
        except httpx.HTTPError as exc:
            raise VisionUnavailable(f"network error: {exc}") from exc
        if resp.status_code >= 400:
            detail = resp.text[:300] if resp.content else ""
            raise VisionUnavailable(
                f"{method} {path} -> {resp.status_code}: {detail}", status=resp.status_code
            )
        return resp.json()

    async def create_stream(self) -> dict:
        return await self._request("POST", "/streams", json_body={})

    async def get_stream(self, stream_id: str) -> dict:
        return await self._request("GET", f"/streams/{stream_id}")

    async def keepalive(self, stream_id: str) -> dict:
        return await self._request("POST", f"/streams/{stream_id}/keepalive")

    async def delete_stream(self, stream_id: str) -> None:
        await self._request("DELETE", f"/streams/{stream_id}")

    async def chat(self, body: dict, *, timeout: float) -> dict:
        return await self._request("POST", "/chat/completions", json_body=body, timeout=timeout)

    async def aclose(self) -> None:
        await self._http.aclose()


class RealtimeVision:
    def __init__(
        self,
        room: rtc.Room,
        participant: str | rtc.RemoteParticipant,
        api_key: str,
        *,
        model: str | None = None,
        instructions: str | None = None,
        track_source: str = "camera",
        max_fps: float = 10.0,
        ingest: str = "auto",
        max_output_tokens: int = 256,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        if ingest not in ("auto", "republish"):
            raise ValueError(f"ingest must be 'auto' or 'republish' (got {ingest!r}); "
                             "'room_join' activates when backend support ships")
        if track_source not in _TRACK_SOURCES:
            raise ValueError(f"track_source must be one of {sorted(_TRACK_SOURCES)}")
        self._room = room
        self._identity = participant if isinstance(participant, str) else participant.identity
        self._model = model or DEFAULT_MODEL
        self._instructions = instructions
        self._source = _TRACK_SOURCES[track_source]
        self._min_frame_interval = 1.0 / max_fps
        self._max_output_tokens = max_output_tokens
        self._api = _Api(api_key, base_url)

        self._stream_id: str | None = None
        self._first_frame_epoch_ms: float | None = None
        self._publish_room: rtc.Room | None = None
        self._video_source: rtc.VideoSource | None = None
        self._bridge_task: asyncio.Task | None = None
        self._keepalive_task: asyncio.Task | None = None
        self._watch_task: asyncio.Task | None = None
        self._start_lock = asyncio.Lock()
        self._closed = False

        room.on("track_subscribed", self._on_track_subscribed)
        room.on("disconnected", self._on_disconnected)
        for p in room.remote_participants.values():
            if p.identity != self._identity:
                continue
            for pub in p.track_publications.values():
                if self._matches(pub) and pub.track is not None:
                    asyncio.create_task(self._start_bridge(pub.track))

    @property
    def stream_id(self) -> str | None:
        """The Overshoot stream id, once the first matching track has been captured."""
        return self._stream_id

    # ---------- track selection / bridge ----------

    def _matches(self, pub: rtc.RemoteTrackPublication) -> bool:
        return pub.kind == rtc.TrackKind.KIND_VIDEO and pub.source == self._source

    def _on_track_subscribed(self, track: rtc.Track, pub: rtc.RemoteTrackPublication,
                             participant: rtc.RemoteParticipant) -> None:
        if not self._closed and participant.identity == self._identity and self._matches(pub):
            asyncio.create_task(self._start_bridge(track))

    def _on_disconnected(self) -> None:
        asyncio.create_task(self.aclose())

    async def _start_bridge(self, track: rtc.Track) -> None:
        async with self._start_lock:
            if self._closed:
                return
            if self._stream_id is None:
                created = await self._api.create_stream()
                self._stream_id = created["id"]
                publish = created["publish"]
                self._publish_room = rtc.Room()
                await self._publish_room.connect(publish["url"], publish["token"])
                ttl = created.get("ttl_seconds") or 120
                self._keepalive_task = asyncio.create_task(self._keepalive_loop(ttl))
                log.info("overshoot stream started stream_id=%s", self._stream_id)
            if self._bridge_task is not None:
                self._bridge_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._bridge_task
            self._bridge_task = asyncio.create_task(self._bridge_loop(track))

    async def _bridge_loop(self, track: rtc.Track) -> None:
        # capacity=1: always forward the freshest frame, never buffer latency.
        stream = rtc.VideoStream(track, capacity=1)
        last_sent = 0.0
        try:
            async for event in stream:
                now = time.monotonic()
                if now - last_sent < self._min_frame_interval:
                    continue
                last_sent = now
                frame = event.frame
                if self._video_source is None:
                    self._video_source = await self._publish(frame.width, frame.height)
                self._video_source.capture_frame(frame)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("frame bridge stopped stream_id=%s", self._stream_id)
        finally:
            with suppress(Exception):
                await stream.aclose()

    async def _publish(self, width: int, height: int) -> rtc.VideoSource:
        assert self._publish_room is not None
        source = rtc.VideoSource(width, height)
        local = rtc.LocalVideoTrack.create_video_track("overshoot-bridge", source)
        await self._publish_room.local_participant.publish_track(
            local, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
        )
        return source

    async def _keepalive_loop(self, ttl_seconds: int) -> None:
        stream_id = self._stream_id
        assert stream_id is not None
        interval = max(10.0, ttl_seconds / 3)
        while not self._closed:
            await asyncio.sleep(interval)
            try:
                resp = await self._api.keepalive(stream_id)
                interval = max(10.0, (resp.get("ttl_seconds") or ttl_seconds) / 3)
            except VisionUnavailable as exc:
                if exc.status == 404:
                    log.warning("stream gone, stopping keepalive stream_id=%s", self._stream_id)
                    return
                log.warning("keepalive failed (will retry): %s", exc)

    # ---------- inference ----------

    async def _first_frame_base(self) -> float:
        """Epoch ms of the stream's first frame; cached (it never changes)."""
        if self._first_frame_epoch_ms is None:
            assert self._stream_id is not None
            status = await self._api.get_stream(self._stream_id)
            base = status.get("first_frame_at_ms")
            if base is None:
                raise VisionUnavailable("stream has no frames yet")
            self._first_frame_epoch_ms = float(base)
        return self._first_frame_epoch_ms

    def _ovs_url(self, window_ms: int, start_ms: float | None, end_ms: float | None,
                 sample_fps: float | None) -> tuple[str, str]:
        base = f"ovs://streams/{self._stream_id}"
        if start_ms is not None:
            params = [f"start_timestamp_ms={int(start_ms)}"]
            if end_ms is not None:
                params.append(f"end_timestamp_ms={int(end_ms)}")
            if sample_fps is not None:
                params.append(f"max_fps={sample_fps}")
            return "video_url", f"{base}?{'&'.join(params)}"
        if window_ms > 0:
            params = [f"start_offset_ms=-{int(window_ms)}"]
            if sample_fps is not None:
                params.append(f"max_fps={sample_fps}")
            return "video_url", f"{base}?{'&'.join(params)}"
        return "image_url", f"{base}?frame_index=-1"

    async def ask(
        self,
        question: str,
        *,
        window_ms: int = 0,
        start_ms: float | None = None,
        end_ms: float | None = None,
        schema: Any = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        sample_fps: float | None = None,
        timeout: float = 10.0,
        extra: dict | None = None,
    ) -> Any:
        """One visual question. Returns text, or parsed data when schema is given.

        Addressing: default = latest frame; window_ms = trailing window;
        start_ms/end_ms = absolute epoch-ms window (end_ms omitted means now).
        schema: a JSON-schema dict (returns dict) or pydantic model class (returns instance).
        """
        text, data = await self._ask(question, window_ms, start_ms, end_ms, schema,
                                     model, max_output_tokens, sample_fps, timeout, extra)
        return text if schema is None else data

    async def _ask(self, question, window_ms, start_ms, end_ms, schema, model,
                   max_output_tokens, sample_fps, timeout, extra) -> tuple[str, Any]:
        if self._stream_id is None:
            raise VisionUnavailable("no video track captured yet")
        if start_ms is not None and window_ms:
            raise ValueError("pass either window_ms or start_ms/end_ms, not both")
        if start_ms is not None:
            # Public API is epoch ms; ovs:// timestamp_ms is stream time (ms since
            # the first captured frame), so convert against the stream's base.
            base = await self._first_frame_base()
            start_ms = max(0.0, start_ms - base)
            end_ms = None if end_ms is None else max(0.0, end_ms - base)
        kind, url = self._ovs_url(window_ms, start_ms, end_ms, sample_fps)
        messages: list[dict] = []
        if self._instructions:
            messages.append({"role": "system", "content": self._instructions})
        messages.append({"role": "user", "content": [
            {"type": kind, kind: {"url": url}},
            {"type": "text", "text": question},
        ]})
        body: dict = {
            "model": model or self._model,
            "messages": messages,
            "max_tokens": max_output_tokens or self._max_output_tokens,
        }
        if schema is not None:
            body["response_format"] = _response_format(schema)
        if extra:
            body.update(extra)
        data = await self._api.chat(body, timeout=timeout)
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise VisionUnavailable(f"malformed completion response: {exc}") from exc
        if schema is None:
            return text, None
        return text, _parse_structured(text, schema)

    def as_tool(self, name: str, description: str, *, window_ms: int = 5000,
                timeout: float = 10.0, allow_history: bool = False):
        """ask() as a LiveKit Agents function tool (returns text for the LLM).

        With allow_history, the tool schema gains an optional seconds_ago parameter so
        the LLM can look into the past on its own.
        """
        from livekit.agents.llm import function_tool

        if allow_history:
            async def _look(question: str, seconds_ago: float = 0) -> str:
                if seconds_ago > 0:
                    end = time.time() * 1000 - seconds_ago * 1000
                    return await self.ask(question, start_ms=end - window_ms, end_ms=end,
                                          timeout=timeout)
                return await self.ask(question, window_ms=window_ms, timeout=timeout)
        else:
            async def _look(question: str) -> str:
                return await self.ask(question, window_ms=window_ms, timeout=timeout)

        return function_tool(name=name, description=description)(_look)

    def watch(self, prompt: str, *, interval: float,
              on_result: Callable[[VisionResult], None],
              window_ms: int = 0, schema: Any = None) -> asyncio.Task:
        """Sequential background loop: one ask() per tick, results to on_result.

        Never overlaps requests (a slow request skips ticks), never raises into the
        agent: transient failures retry next tick; fatal 4xx stops the loop with one log.
        """
        if self._watch_task is not None and not self._watch_task.done():
            raise RuntimeError("watch() is already running; cancel it first")

        async def _loop() -> None:
            while not self._closed:
                t0 = time.monotonic()
                try:
                    text, data = await self._ask(prompt, window_ms, None, None, schema,
                                                 None, None, None, max(interval, 10.0), None)
                except VisionUnavailable as exc:
                    if exc.status in _FATAL_STATUSES:
                        log.error("watch stopped on fatal error: %s", exc)
                        return
                    log.debug("watch tick skipped: %s", exc)
                else:
                    try:
                        on_result(VisionResult(text=text, data=data))
                    except Exception:
                        log.exception("watch on_result callback raised")
                await asyncio.sleep(max(0.0, interval - (time.monotonic() - t0)))

        self._watch_task = asyncio.create_task(_loop())
        return self._watch_task

    # ---------- teardown ----------

    async def aclose(self) -> None:
        """Idempotent teardown: stop tasks, disconnect, delete the Overshoot stream."""
        if self._closed:
            return
        self._closed = True
        self._room.off("track_subscribed", self._on_track_subscribed)
        self._room.off("disconnected", self._on_disconnected)
        for task in (self._watch_task, self._bridge_task, self._keepalive_task):
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        if self._publish_room is not None:
            with suppress(Exception):
                await self._publish_room.disconnect()
        if self._stream_id is not None:
            with suppress(VisionUnavailable):
                await self._api.delete_stream(self._stream_id)
        await self._api.aclose()


def _response_format(schema: Any) -> dict:
    if hasattr(schema, "model_json_schema"):
        return {"type": "json_schema", "json_schema": {
            "name": schema.__name__, "schema": schema.model_json_schema()}}
    if isinstance(schema, dict):
        return {"type": "json_schema", "json_schema": {"name": "response", "schema": schema}}
    raise ValueError("schema must be a JSON-schema dict or a pydantic model class")


def _parse_structured(text: str, schema: Any) -> Any:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VisionSchemaError(f"model output is not valid JSON: {exc}") from exc
    if hasattr(schema, "model_validate"):
        try:
            return schema.model_validate(parsed)
        except Exception as exc:
            raise VisionSchemaError(f"model output failed schema validation: {exc}") from exc
    return parsed
