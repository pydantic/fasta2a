"""Integration tests for the streaming endpoint."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx_sse import aconnect_sse
from sse_starlette.sse import AppStatus

from fasta2a import FastA2A, Worker
from fasta2a.broker import InMemoryBroker
from fasta2a.schema import Artifact, Message, TaskIdParams, TaskSendParams
from fasta2a.storage import InMemoryStorage


def make_stream_request(
    text: str, request_id: str | int | None = 'test-req', message_id: str = 'test-msg'
) -> dict[str, Any]:
    """Create a JSON-RPC streaming request with defaults."""
    return {
        'jsonrpc': '2.0',
        'id': request_id,
        'method': 'message/stream',
        'params': {
            'message': {
                'role': 'user',
                'parts': [{'kind': 'text', 'text': text}],
                'messageId': message_id,
                'kind': 'message',
            }
        },
    }


def make_send_request(
    text: str, request_id: str | int | None = 'test-req', message_id: str = 'test-msg'
) -> dict[str, Any]:
    """Create a JSON-RPC send request with defaults."""
    return {
        'jsonrpc': '2.0',
        'id': request_id,
        'method': 'message/send',
        'params': {
            'message': {
                'role': 'user',
                'parts': [{'kind': 'text', 'text': text}],
                'messageId': message_id,
                'kind': 'message',
            }
        },
    }


def make_get_task_request(task_id: str, request_id: str | int | None = 'test-req') -> dict[str, Any]:
    """Create a JSON-RPC get task request with defaults."""
    return {'jsonrpc': '2.0', 'id': request_id, 'method': 'tasks/get', 'params': {'id': task_id}}


async def collect_sse_events(
    client: httpx.AsyncClient,
    request_data: dict[str, Any],
    stop_condition: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    """Collect all SSE events from a streaming request."""
    events: list[dict[str, Any]] = []
    async with aconnect_sse(client, 'POST', '/', json=request_data) as event_source:
        async for sse in event_source.aiter_sse():
            event_data = json.loads(sse.data)
            events.append(event_data)
            if stop_condition and stop_condition(event_data):
                break
    return events


def get_status_updates(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract status update events."""
    return [e for e in events if e.get('result', {}).get('kind') == 'status-update']


def get_artifact_updates(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract artifact update events."""
    return [e for e in events if e.get('result', {}).get('kind') == 'artifact-update']


def get_tasks(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract task events."""
    return [e for e in events if e.get('result', {}).get('kind') == 'task']


Context = list[Message]
"""The shape of the context you store in the storage."""


@pytest.fixture(autouse=True)
def reset_sse_app_status():
    """Reset SSE global state between tests."""
    AppStatus.should_exit_event = None
    yield
    AppStatus.should_exit_event = None


class StreamingWorker(Worker[Context]):
    """A test worker that emits streaming events."""

    async def run_task(self, params: TaskSendParams) -> None:
        task_id = params['id']
        context_id = params['context_id']

        # Load the task
        task = await self.storage.load_task(task_id)
        assert task is not None

        # Update status to working
        await self.storage.update_task(task_id, state='working')

        # Emit initial status update
        await self.broker.send_stream_event(
            task_id,
            {
                'kind': 'status-update',
                'task_id': task_id,
                'context_id': context_id,
                'status': {'state': 'working'},
                'final': False,
            },
        )

        # Simulate some work with incremental updates
        result_parts = ['Hello', ' from', ' streaming', ' worker!']

        for i, part in enumerate(result_parts):
            # Create an artifact part
            artifact: Artifact = {'artifact_id': 'result-1', 'parts': [{'kind': 'text', 'text': part}]}

            # Emit artifact update
            await self.broker.send_stream_event(
                task_id,
                {
                    'kind': 'artifact-update',
                    'task_id': task_id,
                    'context_id': context_id,
                    'artifact': artifact,
                    'append': i > 0,  # Append after first part
                    'last_chunk': i == len(result_parts) - 1,
                },
            )

        # Store the complete artifact
        complete_artifact: Artifact = {
            'artifact_id': 'result-1',
            'parts': [{'kind': 'text', 'text': 'Hello from streaming worker!'}],
        }

        # Update task with final status
        await self.storage.update_task(task_id, state='completed', new_artifacts=[complete_artifact])

        # Emit final status update
        await self.broker.send_stream_event(
            task_id,
            {
                'kind': 'status-update',
                'task_id': task_id,
                'context_id': context_id,
                'status': {'state': 'completed'},
                'final': True,
            },
        )

    async def cancel_task(self, params: TaskIdParams) -> None:
        await self.storage.update_task(params['id'], state='canceled')

    def build_message_history(self, history: list[Message]) -> list[Any]:
        return history

    def build_artifacts(self, result: Any) -> list[Artifact]:
        return []


@pytest_asyncio.fixture(scope='function')
async def streaming_app() -> FastA2A:
    """Create a FastA2A app with streaming enabled and a streaming worker."""
    storage = InMemoryStorage()
    broker = InMemoryBroker()
    worker = StreamingWorker(storage=storage, broker=broker)

    @asynccontextmanager
    async def lifespan(app: FastA2A) -> AsyncIterator[None]:
        async with app.task_manager:
            async with worker.run():
                yield

    app = FastA2A(
        storage=storage,
        broker=broker,
        streaming=True,  # Enable streaming
        lifespan=lifespan,
    )

    return app


# ===== Basic Streaming Functionality =====


@pytest.mark.asyncio
async def test_streaming_endpoint_basic(streaming_app: FastA2A) -> None:
    """Test basic streaming functionality."""
    async with LifespanManager(streaming_app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=streaming_app), base_url='http://test'
        ) as client:
            # Send a streaming request
            request_data = make_stream_request('Test streaming', 'test-1', 'msg-1')
            events_received = await collect_sse_events(client, request_data)

            # Verify we received events
            assert len(events_received) > 0

            # First event should be the task
            first_event = events_received[0]
            assert first_event['jsonrpc'] == '2.0'
            assert first_event['id'] == 'test-1'
            assert 'result' in first_event
            assert first_event['result']['kind'] == 'task'
            assert first_event['result']['status']['state'] == 'submitted'

            # Should have status updates
            status_updates = get_status_updates(events_received)
            assert len(status_updates) >= 2  # At least working and completed

            # Should have artifact updates
            artifact_updates = get_artifact_updates(events_received)
            assert len(artifact_updates) == 4  # 4 parts

            # Last status update should be final
            last_status = status_updates[-1]
            assert last_status['result']['status']['state'] == 'completed'
            assert last_status['result']['final'] is True


# ===== Artifact Streaming =====


@pytest.mark.asyncio
async def test_streaming_endpoint_incremental_artifacts(streaming_app: FastA2A) -> None:
    """Test that artifacts are streamed incrementally."""
    async with LifespanManager(streaming_app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=streaming_app), base_url='http://test'
        ) as client:
            request_data = make_stream_request('Test artifacts', 'test-2', 'msg-2')
            events = await collect_sse_events(client, request_data)
            artifact_events = [e['result'] for e in get_artifact_updates(events)]

            # Verify artifact streaming
            assert len(artifact_events) == 4

            # First artifact should not append
            assert 'append' not in artifact_events[0] or artifact_events[0]['append'] is False
            assert artifact_events[0]['artifact']['parts'][0]['text'] == 'Hello'

            # Subsequent artifacts should append
            assert artifact_events[1]['append'] is True
            assert artifact_events[1]['artifact']['parts'][0]['text'] == ' from'

            assert artifact_events[2]['append'] is True
            assert artifact_events[2]['artifact']['parts'][0]['text'] == ' streaming'

            # Last artifact should be marked as last chunk
            assert artifact_events[3]['append'] is True
            assert artifact_events[3]['artifact']['parts'][0]['text'] == ' worker!'
            assert artifact_events[3]['lastChunk'] is True


# ===== Streaming vs Non-Streaming Comparison =====


@pytest.mark.asyncio
async def test_streaming_vs_non_streaming_endpoints(streaming_app: FastA2A) -> None:
    """Test that both streaming and non-streaming endpoints work."""
    async with LifespanManager(streaming_app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=streaming_app), base_url='http://test'
        ) as client:
            # First, test non-streaming endpoint
            non_streaming_request = make_send_request('Non-streaming test', 'test-3', 'msg-3')

            # Non-streaming should return immediately with task
            response = await client.post('/', json=non_streaming_request)
            assert response.status_code == 200
            data = response.json()
            assert data['result']['kind'] == 'task'
            assert data['result']['status']['state'] == 'submitted'
            task_id = data['result']['id']

            # Check task status with minimal polling
            get_task_request = make_get_task_request(task_id, 'test-4')

            # Poll for completion with short timeout
            for _ in range(10):  # Try up to 10 times (1 second max)
                response = await client.post('/', json=get_task_request)
                assert response.status_code == 200
                data = response.json()
                if data['result']['status']['state'] == 'completed':
                    break
                await asyncio.sleep(0.1)  # Small delay between polls
            else:
                pytest.fail(f'Task did not complete, final state: {data["result"]["status"]["state"]}')

            assert len(data['result'].get('artifacts', [])) == 1


# ===== Agent Card and Capabilities =====


@pytest.mark.asyncio
async def test_agent_card_shows_streaming_capability(streaming_app: FastA2A) -> None:
    """Test that agent card correctly reports streaming capability."""
    async with LifespanManager(streaming_app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=streaming_app), base_url='http://test'
        ) as client:
            response = await client.get('/.well-known/agent.json')
            assert response.status_code == 200

            agent_card = response.json()
            assert agent_card['capabilities']['streaming'] is True
            assert agent_card['capabilities']['pushNotifications'] is False
            assert agent_card['capabilities']['stateTransitionHistory'] is False


@pytest.mark.asyncio
async def test_non_streaming_app():
    """Test app with streaming disabled."""
    storage = InMemoryStorage()
    broker = InMemoryBroker()

    app = FastA2A(
        storage=storage,
        broker=broker,
        streaming=False,  # Streaming disabled
    )

    async with LifespanManager(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            # Check agent card
            response = await client.get('/.well-known/agent.json')
            assert response.status_code == 200

            agent_card = response.json()
            assert agent_card['capabilities']['streaming'] is False


# ===== ID Validation Tests =====


@pytest.mark.asyncio
async def test_streaming_null_id_accepted(streaming_app: FastA2A) -> None:
    """Test streaming endpoint with explicit null ID - should work."""
    async with LifespanManager(streaming_app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=streaming_app), base_url='http://test'
        ) as client:
            # Request with explicit null ID
            request_data = make_stream_request('Test', None, 'msg-null')
            events_received = await collect_sse_events(client, request_data, lambda _: True)  # Stop after first event

            # Verify the event has null ID
            assert len(events_received) > 0
            assert events_received[0]['id'] is None


@pytest.mark.asyncio
async def test_streaming_numeric_id_accepted(streaming_app: FastA2A) -> None:
    """Test streaming endpoint with numeric ID - should work per JSON-RPC spec."""
    async with LifespanManager(streaming_app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=streaming_app), base_url='http://test'
        ) as client:
            # Request with numeric ID
            request_data = make_stream_request('Test', 12345, 'msg-numeric')
            events_received = await collect_sse_events(client, request_data, lambda _: True)  # Stop after first event

            # Verify the event has numeric ID
            assert len(events_received) > 0
            assert events_received[0]['id'] == 12345


@pytest.mark.asyncio
async def test_streaming_large_message(streaming_app: FastA2A) -> None:
    """Test streaming with a large message payload."""
    async with LifespanManager(streaming_app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=streaming_app), base_url='http://test'
        ) as client:
            # Create a large message
            large_text = 'x' * 10000  # 10KB of text
            request_data = make_stream_request(large_text, 'test-large', 'msg-large')
            events_received = await collect_sse_events(client, request_data)

            # Should still process successfully
            assert len(events_received) > 0
            # Check we got a completed status
            status_updates = get_status_updates(events_received)
            last_status = status_updates[-1]
            assert last_status['result']['status']['state'] == 'completed'
