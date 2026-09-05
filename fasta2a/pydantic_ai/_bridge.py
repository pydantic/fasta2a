from __future__ import annotations

import base64
import uuid
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Generic, TypeVar

from pydantic import TypeAdapter

try:
    from pydantic_ai import (
        Agent,
        AudioUrl,
        BinaryContent,
        DocumentUrl,
        ImageUrl,
        ModelMessage,
        ModelRequest,
        ModelRequestPart,
        ModelResponse,
        ModelResponsePart,
        PartDeltaEvent,
        PartStartEvent,
        TextPart,
        TextPartDelta,
        ThinkingPart,
        ToolCallPart,
        UserPromptPart,
        VideoUrl,
    )
    from pydantic_ai._run_context import AgentDepsT
    from pydantic_ai.agent import AbstractAgent, AgentRunResult
    from pydantic_ai.messages import AgentStreamEvent
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
    AgentExtension,
    AgentProvider,
    Artifact,
    Message,
    Part,
    Skill,
    TaskIdParams,
    TaskSendParams,
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
    extensions: list[AgentExtension] | None = None,
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
        extensions=extensions,
        debug=debug,
        routes=routes,
        middleware=middleware,
        exception_handlers=exception_handlers,
        lifespan=lifespan,
    )


class _CannotStream(Exception):
    """The model refused a streamed request before anything had happened."""


@dataclass
class AgentWorker(Worker[list[ModelMessage]], Generic[WorkerOutputT, AgentDepsT]):
    """A worker that uses a pydantic-ai agent to execute tasks."""

    agent: AbstractAgent[AgentDepsT, WorkerOutputT]
    _streaming: bool | None = field(default=None, init=False, repr=False)
    """Whether the agent's model streams: unknown until the first task tries."""

    async def run_task(self, params: TaskSendParams) -> None:
        task = await self.storage.load_task(params['id'])
        if task is None:
            raise ValueError(f'Task {params["id"]} not found')

        if task['status']['state'] != 'submitted':
            raise ValueError(f'Task {params["id"]} has already been processed (state: {task["status"]["state"]})')

        task_id = task['id']
        context_id = task['context_id']
        await self.storage.update_task(task_id, state='working')
        await self.publish_status(task_id, context_id, 'working')

        message_history = await self.storage.load_context(context_id) or []
        message_history.extend(self.build_message_history(task.get('history', [])))

        # The answer streams as chunks of one artifact while the model writes it; the whole
        # artifact follows as the last chunk, under the same id.
        artifact_id = str(uuid.uuid4())
        try:
            result, streamed = await self._run_agent(task_id, context_id, artifact_id, message_history)

            await self.storage.update_context(context_id, result.all_messages())

            a2a_messages: list[Message] = []
            for message in result.new_messages():
                if isinstance(message, ModelRequest):
                    continue
                a2a_parts = self._response_parts_to_a2a(message.parts)
                if a2a_parts:
                    a2a_messages.append(Message(role='agent', parts=a2a_parts, message_id=str(uuid.uuid4())))

            artifacts = self.build_artifacts(result.output)
            if streamed and artifacts:
                artifacts[0]['artifact_id'] = artifact_id
        except Exception:
            await self.storage.update_task(task_id, state='failed')
            raise
        else:
            await self.storage.update_task(
                task_id, state='completed', new_artifacts=artifacts, new_messages=a2a_messages
            )
            for artifact in artifacts:
                await self.publish_artifact(task_id, context_id, artifact)

    async def _run_agent(
        self, task_id: str, context_id: str, artifact_id: str, message_history: list[ModelMessage]
    ) -> tuple[AgentRunResult[WorkerOutputT], bool]:
        """Run the agent, publishing the text it writes as chunks of the answer's artifact.

        The run is streamed when the model streams. A model that does not — a
        `FunctionModel` without a `stream_function`, say — refuses on its first
        request, before anything has happened; the run is then done again without
        streaming, and that is remembered so the next task does not try.

        Returns the run's result and whether any text was streamed.
        """
        if self._streaming is not False:
            try:
                return await self._run_agent_streaming(task_id, context_id, artifact_id, message_history)
            except _CannotStream:
                self._streaming = False
        result = await self.agent.run(message_history=message_history)  # type: ignore
        return result, False

    async def _run_agent_streaming(
        self, task_id: str, context_id: str, artifact_id: str, message_history: list[ModelMessage]
    ) -> tuple[AgentRunResult[WorkerOutputT], bool]:
        streamed = False
        # Whether the run got anywhere: an event received, or a stream completed.
        # A model that cannot stream refuses before either — on entering the
        # stream in some releases, on its first read in others.
        progressed = False
        try:
            async with self.agent.iter(message_history=message_history) as run:  # type: ignore
                async for node in run:
                    if not Agent.is_model_request_node(node):
                        continue
                    async with node.stream(run.ctx) as request_stream:
                        async for event in request_stream:
                            progressed = True
                            delta = _text_delta(event)
                            if delta:
                                streamed = True
                                await self.publish_artifact(
                                    task_id,
                                    context_id,
                                    Artifact(artifact_id=artifact_id, parts=[Part(text=delta)]),
                                    append=True,
                                    last_chunk=False,
                                )
                    progressed = True
        except (NotImplementedError, AssertionError) as exc:
            # `Model.request_stream` raises NotImplementedError when a model does
            # not stream, and FunctionModel asserts — before any request is made.
            # The same errors once the run has progressed are real ones.
            if progressed:
                raise
            raise _CannotStream() from exc
        self._streaming = True
        result = run.result
        if result is None:  # pragma: no cover - the run has been iterated to its end
            raise RuntimeError('The agent run ended without a result')
        return result, streamed

    async def cancel_task(self, params: TaskIdParams) -> None:
        pass

    def build_artifacts(self, result: WorkerOutputT) -> list[Artifact]:
        artifact_id = str(uuid.uuid4())
        part = self._convert_result_to_part(result)
        return [Artifact(artifact_id=artifact_id, name='result', parts=[part])]

    def _convert_result_to_part(self, result: WorkerOutputT) -> Part:
        if isinstance(result, str):
            return Part(text=result)
        output_type = type(result)
        type_adapter = TypeAdapter(output_type)
        data = type_adapter.dump_python(result, mode='json')
        json_schema = type_adapter.json_schema(mode='serialization')
        return Part(data={'result': data}, metadata={'json_schema': json_schema})

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
            if 'text' in part:
                model_parts.append(UserPromptPart(content=part['text']))
            elif 'raw' in part:
                data = base64.b64decode(part['raw'])
                mime_type = part.get('media_type', 'application/octet-stream')
                content = BinaryContent(data=data, media_type=mime_type)
                model_parts.append(UserPromptPart(content=[content]))
            elif 'url' in part:
                url = part['url']
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
            elif 'data' in part:
                raise NotImplementedError('Data parts are not supported yet.')
            else:
                raise ValueError(f'Unsupported part: {part}')
        return model_parts

    def _response_parts_from_a2a(self, parts: list[Part]) -> list[ModelResponsePart]:
        model_parts: list[ModelResponsePart] = []
        for part in parts:
            if 'text' in part:
                model_parts.append(TextPart(content=part['text']))
            elif 'raw' in part or 'url' in part:
                raise NotImplementedError('File parts are not supported yet.')
            elif 'data' in part:
                raise NotImplementedError('Data parts are not supported yet.')
            else:
                raise ValueError(f'Unsupported part: {part}')
        return model_parts

    def _response_parts_to_a2a(self, parts: Sequence[ModelResponsePart]) -> list[Part]:
        a2a_parts: list[Part] = []
        for part in parts:
            if isinstance(part, TextPart):
                a2a_parts.append(Part(text=part.content))
            elif isinstance(part, ThinkingPart):
                a2a_parts.append(
                    Part(
                        text=part.content,
                        metadata={'type': 'thinking', 'thinking_id': part.id, 'signature': part.signature},
                    )
                )
            elif isinstance(part, ToolCallPart):
                pass
        return a2a_parts


def _text_delta(event: AgentStreamEvent) -> str | None:
    """The text a model stream event adds to the answer, if any."""
    if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
        return event.part.content or None
    if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
        return event.delta.content_delta or None
    return None
