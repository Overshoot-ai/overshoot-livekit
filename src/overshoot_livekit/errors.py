"""Typed errors. Agent code needs exactly one except clause: VisionUnavailable."""


class VisionUnavailable(Exception):
    """Vision is temporarily unavailable (rate limit, server error, timeout, no frames yet).

    The agent should degrade gracefully ("I cannot see you right now") and keep running.
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class VisionSchemaError(VisionUnavailable):
    """The model's output did not parse against the requested schema."""
