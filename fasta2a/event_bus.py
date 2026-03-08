"""Event bus for streaming task updates to SSE connections."""

from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
import anyio.abc

from .schema import StreamResponse


class EventBus(ABC):
    """A pub/sub event bus for streaming task events.

    Allows workers to emit events that are delivered to SSE connections.
    """

    @abstractmethod
    @asynccontextmanager
    async def subscribe(self, task_id: str) -> AsyncIterator[anyio.abc.ObjectReceiveStream[StreamResponse]]:
        """Subscribe to events for a task. Yields a receive stream."""
        yield  # type: ignore[misc]

    @abstractmethod
    async def emit(self, task_id: str, event: StreamResponse) -> None:
        """Emit an event to all subscribers for a task."""

    @abstractmethod
    async def close(self, task_id: str) -> None:
        """Close all subscriber streams for a task, signaling end of SSE."""


class InMemoryEventBus(EventBus):
    """An in-memory event bus using anyio memory streams."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[anyio.abc.ObjectSendStream[StreamResponse]]] = defaultdict(list)

    @asynccontextmanager
    async def subscribe(self, task_id: str) -> AsyncIterator[anyio.abc.ObjectReceiveStream[StreamResponse]]:
        """Subscribe to events for a task. Yields a receive stream."""
        send_stream, receive_stream = anyio.create_memory_object_stream[StreamResponse]()
        self._subscribers[task_id].append(send_stream)
        try:
            yield receive_stream
        finally:
            subscribers = self._subscribers.get(task_id)
            if subscribers is not None:
                try:
                    subscribers.remove(send_stream)
                except ValueError:
                    pass
                if not subscribers:
                    del self._subscribers[task_id]
            await send_stream.aclose()
            await receive_stream.aclose()

    async def emit(self, task_id: str, event: StreamResponse) -> None:
        """Emit an event to all subscribers for a task."""
        for send_stream in self._subscribers.get(task_id, []):
            await send_stream.send(event)

    async def close(self, task_id: str) -> None:
        """Close all subscriber streams for a task, signaling end of SSE."""
        for send_stream in self._subscribers.pop(task_id, []):
            await send_stream.aclose()
