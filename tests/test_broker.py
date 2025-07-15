import pytest
import asyncio
from fasta2a.broker import InMemoryBroker
from fasta2a.schema import TaskSendParams

pytestmark = pytest.mark.anyio

@pytest.fixture
def broker():
    return InMemoryBroker()

async def test_send_and_receive_task(broker: InMemoryBroker):
    task_params: TaskSendParams = {
        'id': 'task_123',
        'context_id': 'ctx_abc',
        'message': {
            'role': 'user',
            'parts': [{'kind': 'text', 'text': 'test message'}],
            'kind': 'message',
            'message_id': 'msg_xyz'
        }
    }

    result_queue = asyncio.Queue()

    async def consumer():
        async for op in broker.receive_task_operations():
            await result_queue.put(op)
            break

    async with broker:
        consumer_task = asyncio.create_task(consumer())
        await asyncio.sleep(0.01)

        await broker.run_task(task_params)

        try:
            received_op = await asyncio.wait_for(result_queue.get(), timeout=1)
        finally:
            consumer_task.cancel()
            try:
                await consumer_task
            except asyncio.CancelledError:
                pass

    assert received_op is not None
    assert received_op['operation'] == 'run'
    assert received_op['params']['id'] == 'task_123'