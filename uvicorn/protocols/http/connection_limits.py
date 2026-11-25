from __future__ import annotations

import random
import time

from uvicorn.config import Config

# Each connection's maximum age is shortened by a random fraction up to this, so that
# connections opened together, e.g. right after a deploy, do not all reconnect together.
MAX_AGE_JITTER = 0.1


class ConnectionLimits:
    """Tracks `limit_connection_max_age` and `limit_connection_max_requests` for one connection."""

    def __init__(self, config: Config) -> None:
        self.max_requests = config.limit_connection_max_requests
        self.deadline: float | None = None
        if config.limit_connection_max_age is not None:
            max_age = config.limit_connection_max_age * (1 - random.uniform(0, MAX_AGE_JITTER))
            self.deadline = time.monotonic() + max_age
        self.requests = 0

    def response_started(self) -> bool:
        """Count a response, and return whether the connection should close once it is sent.

        Called when the response headers are written, so they can carry `Connection: close`.
        """
        self.requests += 1
        if self.max_requests is not None and self.requests >= self.max_requests:
            return True
        return self.deadline is not None and time.monotonic() >= self.deadline
