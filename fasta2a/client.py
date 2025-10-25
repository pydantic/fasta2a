from __future__ import annotations as _annotations

import uuid
from typing import Any

import pydantic

from .schema import (
    AgentCard,
    GetTaskRequest,
    GetTaskResponse,
    Message,
    MessageSendConfiguration,
    MessageSendParams,
    SendMessageRequest,
    SendMessageResponse,
    a2a_request_ta,
    send_message_request_ta,
    send_message_response_ta,
)

get_task_response_ta = pydantic.TypeAdapter(GetTaskResponse)

try:
    import httpx
except ImportError as _import_error:
    raise ImportError(
        'httpx is required to use the A2AClient. Please install it with `pip install httpx`.',
    ) from _import_error


class A2AClient:
    """A client for the A2A protocol."""

    def __init__(
        self,
        agent: str | AgentCard,
        http_client: httpx.AsyncClient | None = None,
        fetch_card: bool = False,
        relative_card_path: str | None = None,
    ) -> None:
        self.agent_card = None
        if fetch_card and isinstance(agent, str):
            if relative_card_path is None:
                relative_card_path = "/.well-known/agent-card.json"
            agent_url = agent.rstrip("/") + relative_card_path
            response = httpx.get(agent_url)
            response.raise_for_status()
            agent = AgentCard(**response.json())
            self.agent_card = agent
        base_url = agent if isinstance(agent, str) else agent['url']
        if http_client is None:
            self.http_client = httpx.AsyncClient(base_url=base_url)
        else:
            self.http_client = http_client
            self.http_client.base_url = base_url

    async def send_message(
        self,
        message: Message,
        *,
        metadata: dict[str, Any] | None = None,
        configuration: MessageSendConfiguration | None = None,
    ) -> SendMessageResponse:
        """Send a message using the A2A protocol.

        Returns a JSON-RPC response containing either a result (Task) or an error.
        """
        params = MessageSendParams(message=message)
        if metadata is not None:
            params['metadata'] = metadata
        if configuration is not None:
            params['configuration'] = configuration

        request_id = str(uuid.uuid4())
        payload = SendMessageRequest(jsonrpc='2.0', id=request_id, method='message/send', params=params)
        content = send_message_request_ta.dump_json(payload, by_alias=True)
        response = await self.http_client.post('/', content=content, headers={'Content-Type': 'application/json'})
        self._raise_for_status(response)

        return send_message_response_ta.validate_json(response.content)

    async def get_task(self, task_id: str) -> GetTaskResponse:
        payload = GetTaskRequest(jsonrpc='2.0', id=None, method='tasks/get', params={'id': task_id})
        content = a2a_request_ta.dump_json(payload, by_alias=True)
        response = await self.http_client.post('/', content=content, headers={'Content-Type': 'application/json'})
        self._raise_for_status(response)
        return get_task_response_ta.validate_json(response.content)

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code >= 400:
            raise UnexpectedResponseError(response.status_code, response.text)


class UnexpectedResponseError(Exception):
    """An error raised when an unexpected response is received from the server."""

    def __init__(self, status_code: int, content: str) -> None:
        self.status_code = status_code
        self.content = content
