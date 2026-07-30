"""overshoot-livekit: real-time vision for LiveKit agents, powered by Overshoot."""

from .errors import VisionSchemaError, VisionUnavailable
from .vision import DEFAULT_BASE_URL, DEFAULT_MODEL, RealtimeVision, VisionResult

__version__ = "0.1.0"

__all__ = [
    "RealtimeVision",
    "VisionResult",
    "VisionUnavailable",
    "VisionSchemaError",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "__version__",
]
