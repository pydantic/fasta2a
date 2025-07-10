# tests/test_task_manager.py

import pytest
import asyncio
from typing import Any

from fasta2a.task_manager import TaskManager
from fasta2a.broker import InMemoryBroker
from fasta2a.storage import InMemoryStorage, ContextT
from fasta2a.worker import Worker
from fasta2a.schema import (
    Artifact,
    Message,
    TextPart,
    TaskSendParams,
    TaskIdParams,
    SendMessageRequest,
    GetTaskRequest
)

pytestmark = pytest.mark.anyio

class MockWorker(Worker[ContextT]):
    def __init__(self, storage, broker):
        super().__init__(storage=storage, broker=broker)
        self.executed_tasks = []
        self.canceled_tasks = []
        self.is_running = asyncio.Event()

    async def run_task(self, params: TaskSendParams) -> None:
        self.executed_tasks.append(params['id'])
        await self.storage.update_task(params['id'], state='working')
        await asyncio.sleep(0.1)
        response_message = Message(
            role='agent',
            parts=[TextPart(text=f"Completed task {params['id']}", kind='text')],
            kind='message',
            message_id='response_msg'
        )
        await self.storage.update_task(
            params['id'],
            state='completed',
            new_messages=[response_message]
        )

    async def cancel_task(self, params: TaskIdParams) -> None:
        self.canceled_tasks.append(params['id'])
        await self.storage.update_task(params['id'], state='canceled')

    def build_message_history(self, history: list[Message]) -> list[Any]:
        return []

    def build_artifacts(self, result: Any) -> list[Artifact]:
        return []

    async def _loop(self) -> None:
        self.is_running.set()
        await super()._loop()

@pytest.fixture
def storage() -> InMemoryStorage[ContextT]:
    return InMemoryStorage()

@pytest.fixture
def broker():
    return InMemoryBroker()

@pytest.fixture
def worker(storage, broker):
    return MockWorker(storage=storage, broker=broker)

@pytest.fixture
def task_manager(storage, broker):
    return TaskManager(storage=storage, broker=broker)

async def test_task_lifecycle(task_manager: TaskManager, worker: MockWorker, storage: InMemoryStorage):
    async with task_manager, worker.run():
        await worker.is_running.wait()
        send_request = SendMessageRequest(
            jsonrpc='2.0',
            id='req1',
            method='message/send',
            params={
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': 'Please run a test task.'}],
                    'kind': 'message',
                    'message_id': 'user_msg1'
                }
            }
        )
        response = await task_manager.send_message(send_request)
        task_id = response['result']['id']
        await asyncio.sleep(0.2)
        assert task_id in worker.executed_tasks
        get_request = GetTaskRequest(
            jsonrpc='2.0',
            id='req2',
            method='tasks/get',
            params={'id': task_id}
        )
        get_response = await task_manager.get_task(get_request)
        assert get_response['result']['status']['state'] == 'completed'
        assert len(get_response['result']['history']) == 2
        assert get_response['result']['history'][-1]['role'] == 'agent'
