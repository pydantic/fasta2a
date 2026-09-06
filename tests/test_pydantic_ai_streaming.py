"""The pydantic-ai bridge streams: the answer arrives as artifact chunks, and the stream ends."""

from __future__ import annotations as _annotations

import sys
import uuid

import anyio
import httpx
import pytest
from asgi_lifespan import LifespanManager

try:
    import pydantic_ai  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ModuleNotFoundError as e:
    # Skip only when pydantic-ai itself is absent (Python 3.9, where the extra
    # does not install); a module missing *under* it is a broken environment.
    if e.name == 'pydantic_ai':
        pytest.skip('pydantic-ai-slim required (Python 3.10+)', allow_module_level=True)
    raise

from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from fasta2a.client import A2AClient
from fasta2a.pydantic_ai import agent_to_a2a
from fasta2a.schema import Message, Part, StreamResponse

pytestmark = [
    pytest.mark.skipif(sys.version_info < (3, 10), reason='pydantic-ai-slim requires 3.10+'),
    pytest.mark.anyio,
]


async def test_stream_message_streams_the_answer_and_ends():
    app = agent_to_a2a(Agent(TestModel(custom_output_text='hello streaming world')))
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = A2AClient(http_client=http_client)
            message = Message(role='user', parts=[Part(text='hi')], message_id=str(uuid.uuid4()))
            events: list[StreamResponse] = [
                response['result'] async for response in client.stream_message(message) if 'result' in response
            ]

    # The stream ended on its own: the task first, the final state last.
    assert 'task' in events[0]
    assert events[0]['task']['status']['state'] == 'submitted'
    states = [event['status_update']['status']['state'] for event in events if 'status_update' in event]
    assert states == ['working', 'completed']
    assert 'status_update' in events[-1]

    # The answer streamed as chunks of one artifact, then came whole as the last chunk of it.
    chunks = [event['artifact_update'] for event in events if 'artifact_update' in event]
    streamed = [chunk for chunk in chunks if chunk.get('append')]
    assert streamed, 'the answer was not streamed'
    assert all(chunk.get('last_chunk') is False for chunk in streamed)
    assert ''.join(part.get('text', '') for chunk in streamed for part in chunk['artifact']['parts']) == (
        'hello streaming world'
    )
    final = chunks[-1]
    assert final.get('append') is False
    assert final.get('last_chunk') is True
    assert [part.get('text') for part in final['artifact']['parts']] == ['hello streaming world']
    assert {chunk['artifact']['artifact_id'] for chunk in chunks} == {final['artifact']['artifact_id']}


def _plain_answer(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content='plain answer')])


async def test_a_model_that_cannot_stream_still_answers_and_the_stream_ends():
    # A FunctionModel without a `stream_function` refuses streamed requests.
    app = agent_to_a2a(Agent(FunctionModel(_plain_answer)))
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = A2AClient(http_client=http_client)
            message = Message(role='user', parts=[Part(text='hi')], message_id=str(uuid.uuid4()))
            with anyio.fail_after(10):
                events: list[StreamResponse] = [
                    response['result'] async for response in client.stream_message(message) if 'result' in response
                ]

    states = [event['status_update']['status']['state'] for event in events if 'status_update' in event]
    assert states == ['working', 'completed']
    chunks = [event['artifact_update'] for event in events if 'artifact_update' in event]
    assert [chunk.get('append') for chunk in chunks] == [False]
    assert [part.get('text') for part in chunks[0]['artifact']['parts']] == ['plain answer']
