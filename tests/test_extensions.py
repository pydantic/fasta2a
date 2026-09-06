"""Extensions: what the card declares, what a request activates, and what the worker sees."""

from __future__ import annotations as _annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from fasta2a import A2A_EXTENSIONS_HEADER, AgentExtension, FastA2A, activated_extensions
from fasta2a.broker import InMemoryBroker
from fasta2a.extensions import (
    ACTIVATED_EXTENSIONS_KEY,
    format_extensions_header,
    missing_required_extensions,
    parse_extensions_header,
    select_activated_extensions,
)
from fasta2a.schema import (
    Artifact,
    Message,
    StreamResponse,
    TaskIdParams,
    TaskSendParams,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from fasta2a.storage import InMemoryStorage
from fasta2a.worker import Worker

pytestmark = pytest.mark.anyio

TRACE = 'urn:example:trace'
CITATIONS = 'urn:example:citations'
UNKNOWN = 'urn:example:nobody-declared-this'


class RecordingWorker(Worker[Any]):
    """Echoes the message back as an artifact and records the extensions each task ran with."""

    def __init__(self, broker: InMemoryBroker, storage: InMemoryStorage[Any]) -> None:
        super().__init__(broker=broker, storage=storage)
        self.seen: list[list[str]] = []

    async def run_task(self, params: TaskSendParams) -> None:
        self.seen.append(activated_extensions(params))
        await self.storage.update_task(params['id'], state='working')
        artifact = Artifact(artifact_id=str(uuid.uuid4()), parts=params['message']['parts'])
        await self.storage.update_task(params['id'], state='completed', new_artifacts=[artifact])
        # A stream ends when its worker says so: the final status, then the bus closed.
        await self.broker.event_bus.emit(
            params['id'],
            StreamResponse(
                status_update=TaskStatusUpdateEvent(
                    task_id=params['id'], context_id=params['context_id'], status=TaskStatus(state='completed')
                )
            ),
        )
        await self.broker.event_bus.close(params['id'])

    async def cancel_task(self, params: TaskIdParams) -> None:
        pass

    def build_message_history(self, history: list[Message]) -> list[Any]:
        return []

    def build_artifacts(self, result: Any) -> list[Artifact]:
        return []


def build_app(extensions: list[AgentExtension] | None) -> tuple[FastA2A, RecordingWorker]:
    storage: InMemoryStorage[Any] = InMemoryStorage()
    broker = InMemoryBroker()
    worker = RecordingWorker(broker=broker, storage=storage)

    @asynccontextmanager
    async def lifespan(app: FastA2A) -> AsyncIterator[None]:
        async with app.task_manager, worker.run():
            yield

    return FastA2A(storage=storage, broker=broker, extensions=extensions, lifespan=lifespan), worker


@asynccontextmanager
async def client_for(app: FastA2A) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://testclient') as client:
            yield client


def send_message_request(method: str = 'message/send') -> dict[str, Any]:
    return {
        'jsonrpc': '2.0',
        'id': str(uuid.uuid4()),
        'method': method,
        'params': {
            'message': {
                'role': 'user',
                'parts': [{'kind': 'text', 'text': 'hello'}],
                'kind': 'message',
                'messageId': str(uuid.uuid4()),
            }
        },
    }


def test_header_parsing_drops_blanks_and_repeats_and_keeps_order():
    assert parse_extensions_header(None) == []
    assert parse_extensions_header('') == []
    assert parse_extensions_header(f' {CITATIONS} ,{TRACE},, {CITATIONS}') == [CITATIONS, TRACE]
    assert format_extensions_header([CITATIONS, TRACE]) == f'{CITATIONS}, {TRACE}'


def test_activation_is_the_intersection_in_the_clients_order():
    supported = [AgentExtension(uri=TRACE), AgentExtension(uri=CITATIONS, required=True)]
    assert select_activated_extensions(supported, [UNKNOWN, CITATIONS, TRACE]) == [CITATIONS, TRACE]
    assert missing_required_extensions(supported, [TRACE]) == [CITATIONS]
    assert missing_required_extensions(supported, [TRACE, CITATIONS]) == []


def test_activated_extensions_reads_the_metadata_key():
    assert activated_extensions({}) == []
    assert activated_extensions({'metadata': None}) == []
    assert activated_extensions({'metadata': 'not-a-mapping'}) == []
    assert activated_extensions({'metadata': {ACTIVATED_EXTENSIONS_KEY: 'not-a-list'}}) == []
    assert activated_extensions({'metadata': {ACTIVATED_EXTENSIONS_KEY: [TRACE]}}) == [TRACE]


async def test_agent_card_declares_the_extensions():
    app, _ = build_app([AgentExtension(uri=TRACE, description='Emits a trace artifact'), AgentExtension(uri=CITATIONS)])
    async with client_for(app) as client:
        response = await client.get('/.well-known/agent-card.json')
    assert response.status_code == 200
    assert response.json()['capabilities']['extensions'] == [
        {'uri': TRACE, 'description': 'Emits a trace artifact'},
        {'uri': CITATIONS},
    ]


async def test_agent_card_without_extensions_declares_none():
    app, _ = build_app(None)
    async with client_for(app) as client:
        response = await client.get('/.well-known/agent-card.json')
    assert 'extensions' not in response.json()['capabilities']


async def test_a_request_activates_supported_extensions_and_the_worker_sees_them():
    app, worker = build_app([AgentExtension(uri=TRACE), AgentExtension(uri=CITATIONS)])
    async with client_for(app) as client:
        response = await client.post(
            '/', json=send_message_request(), headers={A2A_EXTENSIONS_HEADER: f'{UNKNOWN}, {CITATIONS}'}
        )
    assert response.status_code == 200
    assert 'result' in response.json()
    # Echoed back without the one the agent never declared: that is how the client learns it was ignored.
    assert response.headers[A2A_EXTENSIONS_HEADER] == CITATIONS
    assert worker.seen == [[CITATIONS]]


async def test_no_header_activates_nothing():
    app, worker = build_app([AgentExtension(uri=TRACE)])
    async with client_for(app) as client:
        response = await client.post('/', json=send_message_request())
    assert response.status_code == 200
    assert A2A_EXTENSIONS_HEADER not in response.headers
    assert worker.seen == [[]]


async def test_a_required_extension_left_inactive_is_refused_before_a_task_exists():
    app, worker = build_app([AgentExtension(uri=CITATIONS, required=True), AgentExtension(uri=TRACE)])
    async with client_for(app) as client:
        response = await client.post('/', json=send_message_request(), headers={A2A_EXTENSIONS_HEADER: TRACE})
    assert response.status_code == 200
    body = response.json()
    assert body['error'] == {
        'code': -32600,
        'message': 'Request payload validation error',
        'data': {'missing_required_extensions': [CITATIONS]},
    }
    assert response.headers[A2A_EXTENSIONS_HEADER] == TRACE
    assert worker.seen == []


async def test_a_stream_echoes_the_activated_extensions():
    app, worker = build_app([AgentExtension(uri=TRACE)])
    async with (
        client_for(app) as client,
        client.stream(
            'POST', '/', json=send_message_request('message/stream'), headers={A2A_EXTENSIONS_HEADER: TRACE}
        ) as response,
    ):
        assert response.status_code == 200
        assert response.headers[A2A_EXTENSIONS_HEADER] == TRACE
        events = [line async for line in response.aiter_lines() if line.startswith('data: ')]
    assert events, 'the stream sent nothing'
    assert worker.seen == [[TRACE]]


async def test_tasks_get_is_not_gated_by_required_extensions():
    app, _ = build_app([AgentExtension(uri=CITATIONS, required=True)])
    async with client_for(app) as client:
        response = await client.post(
            '/', json={'jsonrpc': '2.0', 'id': 1, 'method': 'tasks/get', 'params': {'id': 'nope'}}
        )
    assert response.json()['error']['code'] == -32001


def test_agent_to_a2a_forwards_extensions():
    try:
        import pydantic_ai  # noqa: F401  # pyright: ignore[reportUnusedImport]
    except ModuleNotFoundError as e:
        if e.name == 'pydantic_ai':
            pytest.skip('pydantic-ai-slim required (Python 3.10+)')
        raise
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    from fasta2a.pydantic_ai import agent_to_a2a

    app = agent_to_a2a(Agent(TestModel()), extensions=[AgentExtension(uri=TRACE)])
    assert app.extensions == [AgentExtension(uri=TRACE)]
