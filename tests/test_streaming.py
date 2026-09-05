from __future__ import annotations as _annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
import httpx
import pytest
from asgi_lifespan import LifespanManager

from fasta2a.applications import FastA2A
from fasta2a.broker import InMemoryBroker
from fasta2a.client import A2AClient
from fasta2a.schema import (
    Artifact,
    Message,
    Part,
    StreamResponse,
    TaskSendParams,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from fasta2a.storage import InMemoryStorage
from fasta2a.worker import Worker

pytestmark = pytest.mark.anyio


class EchoWorker(Worker[Any]):
    """A simple worker that echoes the input message as an artifact."""

    async def run_task(self, params: TaskSendParams) -> None:
        task_id = params['id']
        context_id = params['context_id']

        # Emit a "working" status update
        await self.broker.event_bus.emit(
            task_id,
            StreamResponse(
                status_update=TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state='working'),
                )
            ),
        )

        # Update storage to working
        await self.storage.update_task(task_id, state='working')

        # Create an artifact echoing the input
        input_parts = params['message']['parts']
        artifact = Artifact(artifact_id=str(uuid.uuid4()), parts=input_parts)
        await self.storage.update_task(task_id, state='completed', new_artifacts=[artifact])

        # Emit completed status
        await self.broker.event_bus.emit(
            task_id,
            StreamResponse(
                status_update=TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state='completed'),
                )
            ),
        )
        await self.broker.event_bus.close(task_id)

    async def cancel_task(self, params: Any) -> None:
        pass

    def build_message_history(self, history: list[Message]) -> list[Any]:
        return []

    def build_artifacts(self, result: Any) -> list[Artifact]:
        return []


class SilentWorker(Worker[Any]):
    """A worker that writes storage and says nothing on the event bus."""

    async def run_task(self, params: TaskSendParams) -> None:
        await self.storage.update_task(params['id'], state='working')
        artifact = Artifact(artifact_id=str(uuid.uuid4()), parts=params['message']['parts'])
        await self.storage.update_task(params['id'], state='completed', new_artifacts=[artifact])

    async def cancel_task(self, params: Any) -> None:
        pass

    def build_message_history(self, history: list[Message]) -> list[Any]:
        return []

    def build_artifacts(self, result: Any) -> list[Artifact]:
        return []


class InputRequiredWorker(SilentWorker):
    """A worker that stops to ask the client something."""

    async def run_task(self, params: TaskSendParams) -> None:
        await self.storage.update_task(params['id'], state='input-required')


class FailingWorker(SilentWorker):
    """A worker whose task blows up."""

    async def run_task(self, params: TaskSendParams) -> None:
        raise RuntimeError('no can do')


@asynccontextmanager
async def create_streaming_app(worker_type: type[Worker[Any]] = EchoWorker) -> AsyncIterator[httpx.AsyncClient]:
    broker = InMemoryBroker()
    storage = InMemoryStorage()
    worker = worker_type(broker=broker, storage=storage)

    app = FastA2A(storage=storage, broker=broker)

    @asynccontextmanager
    async def lifespan(app: FastA2A) -> AsyncIterator[None]:
        async with app.task_manager:
            async with worker.run():
                yield

    app = FastA2A(storage=storage, broker=broker, lifespan=lifespan)

    async with LifespanManager(app=app) as manager:
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport, base_url='http://testclient') as client:
            yield client


async def test_stream_message():
    async with create_streaming_app() as http_client:
        client = A2AClient(http_client=http_client)
        client.http_client.base_url = 'http://testclient'

        message = Message(
            role='user',
            parts=[Part(text='Hello, world!')],
            message_id=str(uuid.uuid4()),
        )

        events: list[StreamResponse] = []
        async for response in client.stream_message(message):
            if 'result' in response:
                events.append(response['result'])

        # Should have: initial task, working status, completed status
        assert len(events) == 3

        # First event: initial task state (submitted)
        assert 'task' in events[0]
        assert events[0]['task']['status']['state'] == 'submitted'

        # Second event: working status update
        assert 'status_update' in events[1]
        assert events[1]['status_update']['status']['state'] == 'working'

        # Third event: completed status update
        assert 'status_update' in events[2]
        assert events[2]['status_update']['status']['state'] == 'completed'


async def test_stream_ends_when_the_worker_only_writes_storage():
    async with create_streaming_app(SilentWorker) as http_client:
        client = A2AClient(http_client=http_client)
        client.http_client.base_url = 'http://testclient'

        message = Message(role='user', parts=[Part(text='Hello, world!')], message_id=str(uuid.uuid4()))
        events: list[StreamResponse] = []
        async for response in client.stream_message(message):
            if 'result' in response:
                events.append(response['result'])

    # The task, then its final state read back from storage — and the stream is over.
    assert len(events) == 2
    assert 'task' in events[0]
    assert events[0]['task']['status']['state'] == 'submitted'
    assert 'status_update' in events[1]
    assert events[1]['status_update']['status']['state'] == 'completed'


async def test_stream_ends_failed_when_the_worker_raises():
    async with create_streaming_app(FailingWorker) as http_client:
        client = A2AClient(http_client=http_client)
        client.http_client.base_url = 'http://testclient'

        message = Message(role='user', parts=[Part(text='Hello, world!')], message_id=str(uuid.uuid4()))
        events: list[StreamResponse] = []
        async for response in client.stream_message(message):
            if 'result' in response:
                events.append(response['result'])

    assert len(events) == 2
    assert 'status_update' in events[1]
    assert events[1]['status_update']['status']['state'] == 'failed'


async def test_a_task_waiting_on_the_client_ends_its_stream_and_resubscribe_returns_at_once():
    async with create_streaming_app(InputRequiredWorker) as http_client:
        client = A2AClient(http_client=http_client)
        client.http_client.base_url = 'http://testclient'

        message = Message(role='user', parts=[Part(text='Hello, world!')], message_id=str(uuid.uuid4()))
        events: list[StreamResponse] = []
        async for response in client.stream_message(message):
            if 'result' in response:
                events.append(response['result'])

        assert len(events) == 2
        assert 'task' in events[0]
        assert 'status_update' in events[1]
        assert events[1]['status_update']['status']['state'] == 'input-required'

        # The worker closed that stream; resubscribing must not wait for events that will not come.
        resubscribe = {
            'jsonrpc': '2.0',
            'id': '2',
            'method': 'tasks/resubscribe',
            'params': {'id': events[0]['task']['id']},
        }
        with anyio.fail_after(5):
            async with http_client.stream('POST', '/', json=resubscribe) as response:
                lines = [line async for line in response.aiter_lines() if line.startswith('data: ')]
    assert len(lines) == 1
    assert json.loads(lines[0][6:])['result']['task']['status']['state'] == 'input-required'
