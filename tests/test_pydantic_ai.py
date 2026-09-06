"""Integration tests for the Pydantic AI bridge submodule (`fasta2a.pydantic_ai`).

Ported from `tests/test_a2a.py` in the [pydantic-ai
repo](https://github.com/pydantic/pydantic-ai), where these tests originally exercised
`Agent.to_a2a()`. With the bridge code now hosted upstream here, this file owns the
A2A integration test surface for the Pydantic AI workflow.

Skipped on Python < 3.10 because `pydantic-ai-slim` requires 3.10+.
"""

# `fasta2a.pydantic_ai._bridge` raises ImportError when pydantic-ai is missing, so pyright sees an
# Unknown branch on every call from this file. Silencing two Unknown-related rules at the file level
# rather than scattering per-line ignores across 11 test bodies.
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations as _annotations

import base64
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
import httpx
import pytest
from asgi_lifespan import LifespanManager
from inline_snapshot import snapshot

try:
    import pydantic_ai  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ModuleNotFoundError as e:
    # Skip only when pydantic-ai itself is absent (Python 3.9, where the extra
    # does not install); a module missing *under* it is a broken environment.
    if e.name == 'pydantic_ai':
        pytest.skip('pydantic-ai-slim required (Python 3.10+)', allow_module_level=True)
    raise

# `dirty_equals` matchers (`IsStr`, `IsDatetime`, `IsNow`) report `__eq__`-based equivalence but pyright
# can't see they're meant to substitute for `str`/`datetime` in equality contexts. Mirror the stubs trick
# used in `pydantic-ai/tests/conftest.py` to satisfy strict typecheck.
if TYPE_CHECKING:

    def IsStr(*args: Any, **kwargs: Any) -> str: ...
    def IsDatetime(*args: Any, **kwargs: Any) -> datetime: ...
    def IsNow(*args: Any, **kwargs: Any) -> datetime: ...
else:
    from dirty_equals import IsDatetime, IsNow, IsStr

from pydantic import BaseModel
from pydantic_ai import (
    Agent,
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart as PydanticAITextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from fasta2a.applications import FastA2A
from fasta2a.client import A2AClient
from fasta2a.pydantic_ai import agent_to_a2a
from fasta2a.schema import Message, Part, SendMessageResponse, Task
from fasta2a.storage import InMemoryStorage

pytestmark = [
    pytest.mark.skipif(sys.version_info < (3, 10), reason='pydantic-ai-slim requires 3.10+'),
    pytest.mark.anyio,
]


def task_from_response(response: SendMessageResponse) -> Task:
    """Unwrap the `Task` from a successful `message/send` response."""
    assert 'result' in response
    assert 'task' in response['result']
    return response['result']['task']


@pytest.fixture(scope='session')
def image_content() -> BinaryContent:
    return BinaryContent(
        data=(Path(__file__).parent / 'assets' / 'tiny.jpg').read_bytes(),
        media_type='image/jpeg',
    )


def return_string(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    assert info.output_tools is not None
    args_json = '{"response": ["foo", "bar"]}'
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])


model = FunctionModel(return_string)


class UserProfile(BaseModel):
    name: str
    age: int
    email: str


def return_pydantic_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    assert info.output_tools is not None
    args_json = '{"name": "John Doe", "age": 30, "email": "john@example.com"}'
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])


pydantic_model = FunctionModel(return_pydantic_model)


async def test_pydantic_model_output():
    """Pydantic model outputs carry metadata including JSON schema."""
    agent = Agent(model=pydantic_model, output_type=UserProfile)
    app: FastA2A = agent_to_a2a(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[Part(text='Get user profile')],
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = task_from_response(response)

            task_id = result['id']

            while task := await a2a_client.get_task(task_id):  # pragma: no branch
                if 'result' in task and task['result']['status']['state'] == 'completed':
                    result = task['result']
                    break
                await anyio.sleep(0.1)

            assert 'artifacts' in result
            assert len(result['artifacts']) == 1
            artifact = result['artifacts'][0]

            assert 'data' in artifact['parts'][0]
            assert artifact['parts'][0]['data'] == {
                'result': {'name': 'John Doe', 'age': 30, 'email': 'john@example.com'}
            }

            metadata = artifact['parts'][0].get('metadata')
            assert metadata is not None

            assert metadata['json_schema'] == snapshot(
                {
                    'properties': {
                        'name': {'title': 'Name', 'type': 'string'},
                        'age': {'title': 'Age', 'type': 'integer'},
                        'email': {'title': 'Email', 'type': 'string'},
                    },
                    'required': ['name', 'age', 'email'],
                    'title': 'UserProfile',
                    'type': 'object',
                }
            )

            assert result.get('history') == snapshot(
                [
                    {
                        'role': 'user',
                        'parts': [{'text': 'Get user profile'}],
                        'message_id': IsStr(),
                        'context_id': IsStr(),
                        'task_id': IsStr(),
                    }
                ]
            )


async def test_runtime_error_without_lifespan():
    agent = Agent(model=model, output_type=tuple[str, str])
    app: FastA2A = agent_to_a2a(agent)

    transport = httpx.ASGITransport(app)
    async with httpx.AsyncClient(transport=transport) as http_client:
        a2a_client = A2AClient(http_client=http_client)

        message = Message(
            role='user',
            parts=[Part(text='Hello, world!')],
            message_id=str(uuid.uuid4()),
        )

        with pytest.raises(RuntimeError, match='TaskManager was not properly initialized.'):
            await a2a_client.send_message(message=message)


async def test_simple():
    agent = Agent(model=model, output_type=tuple[str, str])
    app: FastA2A = agent_to_a2a(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[Part(text='Hello, world!')],
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = task_from_response(response)
            assert result == snapshot(
                {
                    'id': IsStr(),
                    'context_id': IsStr(),
                    'status': {'state': 'submitted', 'timestamp': IsDatetime(iso_string=True)},
                    'history': [
                        {
                            'role': 'user',
                            'parts': [{'text': 'Hello, world!'}],
                            'message_id': IsStr(),
                            'context_id': IsStr(),
                            'task_id': IsStr(),
                        }
                    ],
                }
            )

            task_id = result['id']

            while task := await a2a_client.get_task(task_id):  # pragma: no branch
                if 'result' in task and task['result']['status']['state'] == 'completed':
                    break
                await anyio.sleep(0.1)

            assert task == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': None,
                    'result': {
                        'id': IsStr(),
                        'context_id': IsStr(),
                        'status': {'state': 'completed', 'timestamp': IsDatetime(iso_string=True)},
                        'history': [
                            {
                                'role': 'user',
                                'parts': [{'text': 'Hello, world!'}],
                                'message_id': IsStr(),
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            }
                        ],
                        'artifacts': [
                            {
                                'artifact_id': IsStr(),
                                'name': 'result',
                                'parts': [
                                    {
                                        'metadata': {'json_schema': {'items': {}, 'type': 'array'}},
                                        'data': {'result': ['foo', 'bar']},
                                    }
                                ],
                            }
                        ],
                    },
                }
            )


async def test_file_message_with_file():
    agent = Agent(model=model, output_type=tuple[str, str])
    app: FastA2A = agent_to_a2a(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[
                    Part(url='https://example.com/file.txt', media_type='text/plain'),
                ],
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = task_from_response(response)
            assert result == snapshot(
                {
                    'id': IsStr(),
                    'context_id': IsStr(),
                    'status': {'state': 'submitted', 'timestamp': IsDatetime(iso_string=True)},
                    'history': [
                        {
                            'role': 'user',
                            'parts': [{'url': 'https://example.com/file.txt', 'media_type': 'text/plain'}],
                            'message_id': IsStr(),
                            'context_id': IsStr(),
                            'task_id': IsStr(),
                        }
                    ],
                }
            )

            task_id = result['id']

            while task := await a2a_client.get_task(task_id):  # pragma: no branch
                if 'result' in task and task['result']['status']['state'] == 'completed':
                    break
                await anyio.sleep(0.1)
            assert task == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': None,
                    'result': {
                        'id': IsStr(),
                        'context_id': IsStr(),
                        'status': {'state': 'completed', 'timestamp': IsDatetime(iso_string=True)},
                        'history': [
                            {
                                'role': 'user',
                                'parts': [{'url': 'https://example.com/file.txt', 'media_type': 'text/plain'}],
                                'message_id': IsStr(),
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            }
                        ],
                        'artifacts': [
                            {
                                'artifact_id': IsStr(),
                                'name': 'result',
                                'parts': [
                                    {
                                        'metadata': {'json_schema': {'items': {}, 'type': 'array'}},
                                        'data': {'result': ['foo', 'bar']},
                                    }
                                ],
                            }
                        ],
                    },
                }
            )


async def test_file_message_with_file_content(image_content: BinaryContent):
    agent = Agent(model=model, output_type=tuple[str, str])
    app: FastA2A = agent_to_a2a(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            base64_image = base64.b64encode(image_content.data).decode('utf-8')
            message = Message(
                role='user',
                parts=[
                    Part(raw=base64_image, media_type=image_content.media_type),
                ],
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = task_from_response(response)
            assert result == snapshot(
                {
                    'id': IsStr(),
                    'context_id': IsStr(),
                    'status': {'state': 'submitted', 'timestamp': IsDatetime(iso_string=True)},
                    'history': [
                        {
                            'role': 'user',
                            'parts': [
                                {
                                    'raw': IsStr(),
                                    'media_type': 'image/jpeg',
                                }
                            ],
                            'message_id': IsStr(),
                            'context_id': IsStr(),
                            'task_id': IsStr(),
                        }
                    ],
                }
            )

            task_id = result['id']

            while task := await a2a_client.get_task(task_id):  # pragma: no branch
                if 'result' in task and task['result']['status']['state'] == 'completed':
                    break
                await anyio.sleep(0.1)
            assert task == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': None,
                    'result': {
                        'id': IsStr(),
                        'context_id': IsStr(),
                        'status': {'state': 'completed', 'timestamp': IsDatetime(iso_string=True)},
                        'history': [
                            {
                                'role': 'user',
                                'parts': [
                                    {
                                        'raw': IsStr(),
                                        'media_type': 'image/jpeg',
                                    }
                                ],
                                'message_id': IsStr(),
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            }
                        ],
                        'artifacts': [
                            {
                                'artifact_id': IsStr(),
                                'name': 'result',
                                'parts': [
                                    {
                                        'metadata': {'json_schema': {'items': {}, 'type': 'array'}},
                                        'data': {'result': ['foo', 'bar']},
                                    }
                                ],
                            }
                        ],
                    },
                }
            )


async def test_file_message_with_data():
    agent = Agent(model=model, output_type=tuple[str, str])
    app: FastA2A = agent_to_a2a(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[Part(data={'foo': 'bar'})],
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = task_from_response(response)
            assert result == snapshot(
                {
                    'id': IsStr(),
                    'context_id': IsStr(),
                    'status': {'state': 'submitted', 'timestamp': IsDatetime(iso_string=True)},
                    'history': [
                        {
                            'role': 'user',
                            'parts': [{'data': {'foo': 'bar'}}],
                            'message_id': IsStr(),
                            'context_id': IsStr(),
                            'task_id': IsStr(),
                        }
                    ],
                }
            )

            task_id = result['id']

            while task := await a2a_client.get_task(task_id):  # pragma: no branch
                if 'result' in task and task['result']['status']['state'] == 'failed':
                    break
                await anyio.sleep(0.1)
            assert task == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': None,
                    'result': {
                        'id': IsStr(),
                        'context_id': IsStr(),
                        'status': {'state': 'failed', 'timestamp': IsDatetime(iso_string=True)},
                        'history': [
                            {
                                'role': 'user',
                                'parts': [{'data': {'foo': 'bar'}}],
                                'message_id': IsStr(),
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            }
                        ],
                    },
                }
            )


async def test_error_handling():
    """Errors during task execution properly update task state to 'failed'."""

    def raise_error(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError('Test error during agent execution')

    error_model = FunctionModel(raise_error)
    agent = Agent(model=error_model, output_type=str)
    app: FastA2A = agent_to_a2a(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[Part(text='Hello, world!')],
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = task_from_response(response)

            task_id = result['id']

            max_attempts = 50  # 5 seconds total
            for _ in range(max_attempts):
                task = await a2a_client.get_task(task_id)
                if 'result' in task and task['result']['status']['state'] == 'failed':
                    break
                await anyio.sleep(0.1)
            else:  # pragma: no cover
                raise AssertionError(f'Task did not fail within {max_attempts * 0.1} seconds')

            assert 'result' in task
            assert task['result']['status']['state'] == 'failed'


async def test_multiple_tasks_same_context():
    """Multiple tasks share the same context_id with accumulated history."""

    messages_received: list[list[ModelMessage]] = []

    def track_messages(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        messages_received.append(messages.copy())
        assert info.output_tools is not None
        args_json = '{"response": ["foo", "bar"]}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    tracking_model = FunctionModel(track_messages)
    agent = Agent(model=tracking_model, output_type=tuple[str, str])
    app: FastA2A = agent_to_a2a(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message1 = Message(
                role='user',
                parts=[Part(text='First message')],
                message_id=str(uuid.uuid4()),
            )
            response1 = await a2a_client.send_message(message=message1)
            assert 'error' not in response1
            assert 'result' in response1
            result1 = task_from_response(response1)

            task1_id = result1['id']
            context_id = result1['context_id']

            while task1 := await a2a_client.get_task(task1_id):  # pragma: no branch
                if 'result' in task1 and task1['result']['status']['state'] == 'completed':
                    result1 = task1['result']
                    break
                await anyio.sleep(0.1)

            assert len(messages_received) == 1
            first_run_history = messages_received[0]
            assert first_run_history == snapshot(
                [
                    ModelRequest(
                        parts=[UserPromptPart(content='First message', timestamp=IsDatetime())],
                        timestamp=IsNow(tz=timezone.utc),
                        run_id=IsStr(),
                        conversation_id=IsStr(),
                    )
                ]
            )

            message2 = Message(
                role='user',
                parts=[Part(text='Second message')],
                context_id=context_id,
                message_id=str(uuid.uuid4()),
            )
            response2 = await a2a_client.send_message(message=message2)
            assert 'error' not in response2
            assert 'result' in response2
            result2 = task_from_response(response2)

            task2_id = result2['id']
            assert task2_id != task1_id
            assert result2['context_id'] == context_id

            while task2 := await a2a_client.get_task(task2_id):  # pragma: no branch
                if 'result' in task2 and task2['result']['status']['state'] == 'completed':
                    break
                await anyio.sleep(0.1)

            assert len(messages_received) == 2
            second_run_history = messages_received[1]
            assert second_run_history[0] == first_run_history[0]

            assert second_run_history == snapshot(
                [
                    ModelRequest(
                        parts=[UserPromptPart(content='First message', timestamp=IsDatetime())],
                        timestamp=IsNow(tz=timezone.utc),
                        run_id=IsStr(),
                        conversation_id=IsStr(),
                    ),
                    ModelResponse(
                        parts=[
                            ToolCallPart(
                                tool_name='final_result', args='{"response": ["foo", "bar"]}', tool_call_id=IsStr()
                            )
                        ],
                        usage=RequestUsage(input_tokens=52, output_tokens=7),
                        model_name='function:track_messages:',
                        timestamp=IsDatetime(),
                        run_id=IsStr(),
                        conversation_id=IsStr(),
                    ),
                    ModelRequest(
                        parts=[
                            ToolReturnPart(
                                tool_name='final_result',
                                content='Final result processed.',
                                tool_call_id=IsStr(),
                                timestamp=IsDatetime(),
                            ),
                            UserPromptPart(content='Second message', timestamp=IsDatetime()),
                        ],
                        timestamp=IsNow(tz=timezone.utc),
                        run_id=IsStr(),
                        conversation_id=IsStr(),
                    ),
                ]
            )


async def test_thinking_response():
    """ModelResponse messages with ThinkingPart are properly serialized to A2A."""

    def return_thinking_response(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        return ModelResponse(
            parts=[
                ThinkingPart(content='Let me think about this...', id='thinking_1'),
                PydanticAITextPart(content="Here's my response"),
            ]
        )

    thinking_model = FunctionModel(return_thinking_response)
    agent = Agent(model=thinking_model, output_type=str)
    app: FastA2A = agent_to_a2a(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[Part(text='Hello, world!')],
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = task_from_response(response)

            task_id = result['id']

            while task := await a2a_client.get_task(task_id):  # pragma: no branch
                if 'result' in task and task['result']['status']['state'] == 'completed':
                    result = task['result']
                    break
                await anyio.sleep(0.1)

            assert result == snapshot(
                {
                    'id': IsStr(),
                    'context_id': IsStr(),
                    'status': {'state': 'completed', 'timestamp': IsDatetime(iso_string=True)},
                    'history': [
                        {
                            'role': 'user',
                            'parts': [{'text': 'Hello, world!'}],
                            'message_id': IsStr(),
                            'context_id': IsStr(),
                            'task_id': IsStr(),
                        },
                        {
                            'role': 'agent',
                            'parts': [
                                {
                                    'metadata': {'type': 'thinking', 'thinking_id': 'thinking_1', 'signature': None},
                                    'text': 'Let me think about this...',
                                },
                                {'text': "Here's my response"},
                            ],
                            'message_id': IsStr(),
                            'context_id': IsStr(),
                            'task_id': IsStr(),
                        },
                    ],
                    'artifacts': [
                        {
                            'artifact_id': IsStr(),
                            'name': 'result',
                            'parts': [{'text': "Here's my response"}],
                        }
                    ],
                }
            )


async def test_multiple_messages():
    agent = Agent(model=model, output_type=tuple[str, str])
    storage = InMemoryStorage()
    app: FastA2A = agent_to_a2a(agent, storage=storage)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[Part(text='Hello, world!')],
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert response == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': IsStr(),
                    'result': {
                        'task': {
                            'id': IsStr(),
                            'context_id': IsStr(),
                            'status': {'state': 'submitted', 'timestamp': IsDatetime(iso_string=True)},
                            'history': [
                                {
                                    'role': 'user',
                                    'parts': [{'text': 'Hello, world!'}],
                                    'message_id': IsStr(),
                                    'context_id': IsStr(),
                                    'task_id': IsStr(),
                                }
                            ],
                        }
                    },
                }
            )

            # Splice an agent message into history before the worker picks up the task.
            assert 'result' in response
            result = task_from_response(response)
            task_id = result['id']
            task = storage.tasks[task_id]
            assert 'history' in task
            task['history'].append(
                Message(
                    role='agent',
                    parts=[Part(text='Whats up?')],
                    message_id=str(uuid.uuid4()),
                )
            )

            response = await a2a_client.get_task(task_id)
            assert response == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': None,
                    'result': {
                        'id': IsStr(),
                        'context_id': IsStr(),
                        'status': {'state': 'submitted', 'timestamp': IsDatetime(iso_string=True)},
                        'history': [
                            {
                                'role': 'user',
                                'parts': [{'text': 'Hello, world!'}],
                                'message_id': IsStr(),
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            },
                            {
                                'role': 'agent',
                                'parts': [{'text': 'Whats up?'}],
                                'message_id': IsStr(),
                            },
                        ],
                    },
                }
            )

            while task := await a2a_client.get_task(task_id):  # pragma: no branch
                if 'result' in task and task['result']['status']['state'] == 'completed':
                    break
                await anyio.sleep(0.1)

            assert task == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': None,
                    'result': {
                        'id': IsStr(),
                        'context_id': IsStr(),
                        'status': {'state': 'completed', 'timestamp': IsDatetime(iso_string=True)},
                        'history': [
                            {
                                'role': 'user',
                                'parts': [{'text': 'Hello, world!'}],
                                'message_id': IsStr(),
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            },
                            {
                                'role': 'agent',
                                'parts': [{'text': 'Whats up?'}],
                                'message_id': IsStr(),
                            },
                        ],
                        'artifacts': [
                            {
                                'artifact_id': IsStr(),
                                'name': 'result',
                                'parts': [
                                    {
                                        'metadata': {'json_schema': {'items': {}, 'type': 'array'}},
                                        'data': {'result': ['foo', 'bar']},
                                    }
                                ],
                            }
                        ],
                    },
                }
            )


async def test_multiple_send_task_messages():
    agent = Agent(model=model, output_type=tuple[str, str])
    storage = InMemoryStorage()
    app: FastA2A = agent_to_a2a(agent, storage=storage)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[Part(text='Hello, world!')],
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert response == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': IsStr(),
                    'result': {
                        'task': {
                            'id': IsStr(),
                            'context_id': IsStr(),
                            'status': {'state': 'submitted', 'timestamp': IsDatetime(iso_string=True)},
                            'history': [
                                {
                                    'role': 'user',
                                    'parts': [{'text': 'Hello, world!'}],
                                    'message_id': IsStr(),
                                    'context_id': IsStr(),
                                    'task_id': IsStr(),
                                }
                            ],
                        }
                    },
                }
            )
            assert 'result' in response
            result = task_from_response(response)
            task_id = result['id']
            context_id = result['context_id']

            while task := await a2a_client.get_task(task_id):  # pragma: no branch
                if 'result' in task and task['result']['status']['state'] == 'completed':
                    result = task['result']
                    break
                await anyio.sleep(0.1)

            assert result == snapshot(
                {
                    'id': IsStr(),
                    'context_id': IsStr(),
                    'status': {'state': 'completed', 'timestamp': IsDatetime(iso_string=True)},
                    'history': [
                        {
                            'role': 'user',
                            'parts': [{'text': 'Hello, world!'}],
                            'message_id': IsStr(),
                            'context_id': IsStr(),
                            'task_id': IsStr(),
                        }
                    ],
                    'artifacts': [
                        {
                            'artifact_id': IsStr(),
                            'name': 'result',
                            'parts': [
                                {
                                    'metadata': {'json_schema': {'items': {}, 'type': 'array'}},
                                    'data': {'result': ['foo', 'bar']},
                                }
                            ],
                        }
                    ],
                }
            )

            message = Message(
                role='user',
                parts=[Part(text='Did you get my first message?')],
                message_id=str(uuid.uuid4()),
                context_id=context_id,
            )
            response = await a2a_client.send_message(message=message)
            assert response == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': IsStr(),
                    'result': {
                        'task': {
                            'id': IsStr(),
                            'context_id': IsStr(),
                            'status': {'state': 'submitted', 'timestamp': IsDatetime(iso_string=True)},
                            'history': [
                                {
                                    'role': 'user',
                                    'parts': [{'text': 'Did you get my first message?'}],
                                    'message_id': IsStr(),
                                    'context_id': IsStr(),
                                    'task_id': IsStr(),
                                }
                            ],
                        }
                    },
                }
            )

            while task := await a2a_client.get_task(task_id):  # pragma: no branch
                if 'result' in task and task['result']['status']['state'] == 'completed':
                    result = task['result']
                    break
                await anyio.sleep(0.1)  # pragma: lax no cover

            assert result == snapshot(
                {
                    'id': IsStr(),
                    'context_id': IsStr(),
                    'status': {'state': 'completed', 'timestamp': IsDatetime(iso_string=True)},
                    'history': [
                        {
                            'role': 'user',
                            'parts': [{'text': 'Hello, world!'}],
                            'message_id': IsStr(),
                            'context_id': IsStr(),
                            'task_id': IsStr(),
                        }
                    ],
                    'artifacts': [
                        {
                            'artifact_id': IsStr(),
                            'name': 'result',
                            'parts': [
                                {
                                    'metadata': {'json_schema': {'items': {}, 'type': 'array'}},
                                    'data': {'result': ['foo', 'bar']},
                                }
                            ],
                        }
                    ],
                }
            )
