from __future__ import annotations

import base64
import uuid
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Generic, TypeVar

from pydantic import TypeAdapter
from typing_extensions import assert_never

try:
    from pydantic_ai import (
        AudioUrl,
        BinaryContent,
        DocumentUrl,
        ImageUrl,
        ModelMessage,
        ModelRequest,
        ModelRequestPart,
        ModelResponse,
        ModelResponsePart,
        TextPart,
        ThinkingPart,
        ToolCallPart,
        UserPromptPart,
        VideoUrl,
    )
    from pydantic_ai._run_context import AgentDepsT
    from pydantic_ai.agent import AbstractAgent
    from pydantic_ai.output import OutputDataT
except ImportError as _e:
    raise ImportError(
        'Please install the `pydantic-ai` package to use `fasta2a.pydantic_ai`, '
        "e.g. `pip install 'fasta2a[pydantic-ai]'`"
    ) from _e

from starlette.middleware import Middleware
from starlette.routing import Route
from starlette.types import ExceptionHandler, Lifespan

from fasta2a.applications import FastA2A
from fasta2a.broker import Broker, InMemoryBroker
from fasta2a.schema import (
    AgentProvider,
    Artifact,
    DataPart,
    Message,
    Part,
    Skill,
    TaskIdParams,
    TaskSendParams,
    TextPart as A2ATextPart,
)
from fasta2a.storage import InMemoryStorage, Storage
from fasta2a.worker import Worker

WorkerOutputT = TypeVar('WorkerOutputT')


@asynccontextmanager
async def worker_lifespan(
    app: FastA2A, worker: Worker, agent: AbstractAgent[AgentDepsT, OutputDataT]
) -> AsyncGenerator[None]:
    """Lifespan that runs the worker during application startup."""
    async with app.task_manager, agent:
        async with worker.run():
            yield


def agent_to_a2a(
    agent: AbstractAgent[AgentDepsT, OutputDataT],
    *,
    storage: Storage | None = None,
    broker: Broker | None = None,
    name: str | None = None,
    url: str = 'http://localhost:8000',
    version: str = '1.0.0',
    description: str | None = None,
    provider: AgentProvider | None = None,
    skills: list[Skill] | None = None,
    debug: bool = False,
    routes: Sequence[Route] | None = None,
    middleware: Sequence[Middleware] | None = None,
    exception_handlers: dict[Any, ExceptionHandler] | None = None,
    lifespan: Lifespan[FastA2A] | None = None,
) -> FastA2A:
    """Create a FastA2A server from a pydantic-ai agent."""
    storage = storage or InMemoryStorage()
    broker = broker or InMemoryBroker()
    worker = AgentWorker(agent=agent, broker=broker, storage=storage)

    lifespan = lifespan or partial(worker_lifespan, worker=worker, agent=agent)

    return FastA2A(
        storage=storage,
        broker=broker,
        name=name or agent.name,
        url=url,
        version=version,
        description=description,
        provider=provider,
        skills=skills,
        debug=debug,
        routes=routes,
        middleware=middleware,
        exception_handlers=exception_handlers,
        lifespan=lifespan,
    )


@dataclass
class AgentWorker(Worker[list[ModelMessage]], Generic[WorkerOutputT, AgentDepsT]):
    """A worker that uses a pydantic-ai agent to execute tasks."""

    agent: AbstractAgent[AgentDepsT, WorkerOutputT]

    async def run_task(self, params: TaskSendParams) -> None:
        task = await self.storage.load_task(params['id'])
        if task is None:
            raise ValueError(f'Task {params["id"]} not found')

        if task['status']['state'] != 'submitted':
            raise ValueError(f'Task {params["id"]} has already been processed (state: {task["status"]["state"]})')

        await self.storage.update_task(task['id'], state='working')

        message_history = await self.storage.load_context(task['context_id']) or []
        message_history.extend(self.build_message_history(task.get('history', [])))

        try:
            result = await self.agent.run(message_history=message_history)  # type: ignore

            await self.storage.update_context(task['context_id'], result.all_messages())

            a2a_messages: list[Message] = []
            for message in result.new_messages():
                if isinstance(message, ModelRequest):
                    continue
                a2a_parts = self._response_parts_to_a2a(message.parts)
                if a2a_parts:
                    a2a_messages.append(
                        Message(role='agent', parts=a2a_parts, kind='message', message_id=str(uuid.uuid4()))
                    )

            artifacts = self.build_artifacts(result.output)
        except Exception:
            await self.storage.update_task(task['id'], state='failed')
            raise
        else:
            await self.storage.update_task(
                task['id'], state='completed', new_artifacts=artifacts, new_messages=a2a_messages
            )

    async def cancel_task(self, params: TaskIdParams) -> None:
        pass

    def build_artifacts(self, result: WorkerOutputT) -> list[Artifact]:
        artifact_id = str(uuid.uuid4())
        part = self._convert_result_to_part(result)
        return [Artifact(artifact_id=artifact_id, name='result', parts=[part])]

    def _convert_result_to_part(self, result: WorkerOutputT) -> Part:
        if isinstance(result, str):
            return A2ATextPart(kind='text', text=result)
        output_type = type(result)
        type_adapter = TypeAdapter(output_type)
        data = type_adapter.dump_python(result, mode='json')
        json_schema = type_adapter.json_schema(mode='serialization')
        return DataPart(kind='data', data={'result': data}, metadata={'json_schema': json_schema})

    def build_message_history(self, history: list[Message]) -> list[ModelMessage]:
        model_messages: list[ModelMessage] = []
        for message in history:
            if message['role'] == 'user':
                model_messages.append(ModelRequest(parts=self._request_parts_from_a2a(message['parts'])))
            else:
                model_messages.append(ModelResponse(parts=self._response_parts_from_a2a(message['parts'])))
        return model_messages

    def _request_parts_from_a2a(self, parts: list[Part]) -> list[ModelRequestPart]:
        model_parts: list[ModelRequestPart] = []
        for part in parts:
            if part['kind'] == 'text':
                model_parts.append(UserPromptPart(content=part['text']))
            elif part['kind'] == 'file':
                file_content = part['file']
                if 'bytes' in file_content:
                    data = base64.b64decode(file_content['bytes'])
                    mime_type = file_content.get('mime_type', 'application/octet-stream')
                    content = BinaryContent(data=data, media_type=mime_type)
                    model_parts.append(UserPromptPart(content=[content]))
                else:
                    url = file_content['uri']
                    for url_cls in (DocumentUrl, AudioUrl, ImageUrl, VideoUrl):
                        content = url_cls(url=url)
                        try:
                            content.media_type
                        except ValueError:
                            continue
                        else:
                            break
                    else:
                        raise ValueError(f'Unsupported file type: {url}')
                    model_parts.append(UserPromptPart(content=[content]))
            elif part['kind'] == 'data':
                raise NotImplementedError('Data parts are not supported yet.')
            else:
                assert_never(part)
        return model_parts

    def _response_parts_from_a2a(self, parts: list[Part]) -> list[ModelResponsePart]:
        model_parts: list[ModelResponsePart] = []
        for part in parts:
            if part['kind'] == 'text':
                model_parts.append(TextPart(content=part['text']))
            elif part['kind'] == 'file':
                raise NotImplementedError('File parts are not supported yet.')
            elif part['kind'] == 'data':
                raise NotImplementedError('Data parts are not supported yet.')
            else:
                assert_never(part)
        return model_parts

    def _response_parts_to_a2a(self, parts: Sequence[ModelResponsePart]) -> list[Part]:
        a2a_parts: list[Part] = []
        for part in parts:
            if isinstance(part, TextPart):
                a2a_parts.append(A2ATextPart(kind='text', text=part.content))
            elif isinstance(part, ThinkingPart):
                a2a_parts.append(
                    A2ATextPart(
                        kind='text',
                        text=part.content,
                        metadata={'type': 'thinking', 'thinking_id': part.id, 'signature': part.signature},
                    )
                )
            elif isinstance(part, ToolCallPart):
                pass
        return a2a_parts
