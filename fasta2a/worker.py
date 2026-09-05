from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic

import anyio
from opentelemetry.trace import get_tracer, use_span
from typing_extensions import assert_never

from .schema import STREAM_ENDING_STATES
from .storage import ContextT, Storage

if TYPE_CHECKING:
    from .broker import Broker, TaskOperation
    from .schema import Artifact, Message, TaskIdParams, TaskSendParams, TaskState

tracer = get_tracer(__name__)


@dataclass
class Worker(ABC, Generic[ContextT]):
    """A worker is responsible for executing tasks.

    While a task runs, a worker reports on the broker's event bus with `publish_status` and
    `publish_artifact`, and a `message/stream` relays what it publishes as it comes. It does not
    have to report the end: once an operation returns, the task's final state is read back from
    storage, published, and the stream closed — and when `run_task` raises, the task is marked
    failed and the same is done — so a worker that only writes storage still ends its stream.
    """

    broker: Broker
    storage: Storage[ContextT]

    @asynccontextmanager
    async def run(self) -> AsyncIterator[None]:
        """Run the worker.

        It connects to the broker, and it makes itself available to receive commands.
        """
        async with anyio.create_task_group() as tg:
            tg.start_soon(self._loop)
            yield
            tg.cancel_scope.cancel()

    async def _loop(self) -> None:
        async for task_operation in self.broker.receive_task_operations():
            await self._handle_task_operation(task_operation)

    async def _handle_task_operation(self, task_operation: TaskOperation) -> None:
        task_id = task_operation['params']['id']
        try:
            with use_span(task_operation['_current_span']):
                with tracer.start_as_current_span(
                    f'{task_operation["operation"]} task', attributes={'logfire.tags': ['fasta2a']}
                ):
                    if task_operation['operation'] == 'run':
                        await self.run_task(task_operation['params'])
                    elif task_operation['operation'] == 'cancel':
                        await self.cancel_task(task_operation['params'])
                    else:
                        assert_never(task_operation)
        except Exception:
            task = await self.storage.update_task(task_id, state='failed')
            await self._end_stream(task_id, task['context_id'], 'failed')
        else:
            await self._publish_final_state(task_id)

    async def publish_status(
        self, task_id: str, context_id: str, state: TaskState, message: Message | None = None
    ) -> None:
        """Tell the task's stream that its state changed, with a message for the client if there is one."""
        from .schema import StreamResponse, TaskStatus, TaskStatusUpdateEvent

        status = TaskStatus(state=state)
        if message is not None:
            status['message'] = message
        await self.broker.event_bus.emit(
            task_id,
            StreamResponse(status_update=TaskStatusUpdateEvent(task_id=task_id, context_id=context_id, status=status)),
        )

    async def publish_artifact(
        self, task_id: str, context_id: str, artifact: Artifact, *, append: bool = False, last_chunk: bool = True
    ) -> None:
        """Send an artifact, or a chunk of one, to the task's stream.

        A chunk is ``append=True`` and, until the last one, ``last_chunk=False``; the client joins
        the chunks that share an ``artifact_id``.
        """
        from .schema import StreamResponse, TaskArtifactUpdateEvent

        await self.broker.event_bus.emit(
            task_id,
            StreamResponse(
                artifact_update=TaskArtifactUpdateEvent(
                    task_id=task_id, context_id=context_id, artifact=artifact, append=append, last_chunk=last_chunk
                )
            ),
        )

    async def _publish_final_state(self, task_id: str) -> None:
        """End the task's stream if the task is over, or waiting on the client.

        The state is read back from storage, so it is whatever the worker left there. A worker
        that already published it and closed the stream has no subscribers left, and this is a
        no-op.
        """
        task = await self.storage.load_task(task_id)
        if task is None:
            return
        state = task['status']['state']
        if state in STREAM_ENDING_STATES:
            await self._end_stream(task_id, task['context_id'], state)

    async def _end_stream(self, task_id: str, context_id: str, state: TaskState) -> None:
        await self.publish_status(task_id, context_id, state)
        await self.broker.event_bus.close(task_id)

    @abstractmethod
    async def run_task(self, params: TaskSendParams) -> None: ...

    @abstractmethod
    async def cancel_task(self, params: TaskIdParams) -> None: ...

    @abstractmethod
    def build_message_history(self, history: list[Message]) -> list[Any]: ...

    @abstractmethod
    def build_artifacts(self, result: Any) -> list[Artifact]: ...
