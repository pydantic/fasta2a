import pytest
from fasta2a.storage import InMemoryStorage, ContextT
from fasta2a.schema import Message, TextPart

pytestmark = pytest.mark.anyio

@pytest.fixture
def storage() -> InMemoryStorage[ContextT]:
    return InMemoryStorage()

@pytest.mark.asyncio
async def test_submit_and_load_task(storage: InMemoryStorage):
    message = Message(
        role='user',
        parts=[TextPart(text='hello', kind='text')],
        kind='message',
        message_id='msg1'
    )
    
    task = await storage.submit_task(context_id="ctx1", message=message)
    assert task is not None
    assert task['id'] is not None
    assert task['status']['state'] == 'submitted'
    assert task['context_id'] == 'ctx1'

    loaded_task = await storage.load_task(task['id'])
    assert loaded_task is not None
    assert loaded_task['id'] == task['id']

@pytest.mark.asyncio
async def test_update_task(storage: InMemoryStorage):
    message = Message(
        role='user',
        parts=[TextPart(text='hello', kind='text')],
        kind='message',
        message_id='msg2'
    )
    task = await storage.submit_task(context_id="ctx2", message=message)
    
    await storage.update_task(task['id'], state='completed')
    
    updated_task = await storage.load_task(task['id'])
    assert updated_task is not None
    assert updated_task['status']['state'] == 'completed'

@pytest.mark.asyncio
async def test_context_storage(storage: InMemoryStorage):
    context = await storage.load_context("ctx3")
    assert context is None

    await storage.update_context("ctx3", ["message1", "message2"])

    loaded_context = await storage.load_context("ctx3")
    assert loaded_context is not None
    assert loaded_context == ["message1", "message2"]