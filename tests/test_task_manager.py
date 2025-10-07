"""Tests for TaskManager class."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Literal
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

from fasta2a.broker import InMemoryBroker
from fasta2a.schema import (
    CancelTaskRequest,
    GetTaskRequest,
    Message,
    MessageSendConfiguration,
    MessageSendParams,
    SendMessageRequest,
    StreamMessageRequest,
    TaskQueryParams,
    TaskSendParams,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from fasta2a.storage import InMemoryStorage
from fasta2a.task_manager import TaskManager

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def storage() -> InMemoryStorage:
    """Create an InMemoryStorage instance."""
    return InMemoryStorage()


@pytest.fixture
def broker() -> InMemoryBroker:
    """Create an InMemoryBroker instance."""
    return InMemoryBroker()


@pytest.fixture
def task_manager_factory(broker: InMemoryBroker, storage: InMemoryStorage) -> TaskManager:
    """Create a TaskManager instance without entering contexts."""
    return TaskManager(broker=broker, storage=storage)


@pytest_asyncio.fixture
async def task_manager(
    broker: InMemoryBroker, storage: InMemoryStorage
) -> AsyncIterator[tuple[TaskManager, InMemoryBroker, InMemoryStorage]]:
    """Create and enter TaskManager with broker contexts.

    Yields:
        tuple: (task_manager, broker, storage) for tests that need access to all components
    """
    tm = TaskManager(broker=broker, storage=storage)
    async with broker:
        async with tm:
            yield tm, broker, storage


# ============================================================================
# Helper Functions
# ============================================================================


def send_message_request(
    message: Message,
    req_id: str = 'req-1',
    configuration: MessageSendConfiguration | None = None,
    metadata: dict[str, Any] | None = None,
) -> SendMessageRequest:
    """Build a SendMessageRequest."""
    params: MessageSendParams = {'message': message}
    if configuration:
        params['configuration'] = configuration
    if metadata:
        params['metadata'] = metadata
    return {
        'jsonrpc': '2.0',
        'id': req_id,
        'method': 'message/send',
        'params': params,
    }


def stream_message_request(
    message: Message,
    req_id: str = 'req-1',
    configuration: MessageSendConfiguration | None = None,
    metadata: dict[str, Any] | None = None,
) -> StreamMessageRequest:
    """Build a StreamMessageRequest."""
    params: MessageSendParams = {'message': message}
    if configuration:
        params['configuration'] = configuration
    if metadata:
        params['metadata'] = metadata
    return {
        'jsonrpc': '2.0',
        'id': req_id,
        'method': 'message/stream',
        'params': params,
    }


def get_task_request(task_id: str, req_id: str = 'req-1', history_length: int | None = None) -> GetTaskRequest:
    """Build a GetTaskRequest."""
    params: TaskQueryParams = {'id': task_id}
    if history_length is not None:
        params['history_length'] = history_length
    return {
        'jsonrpc': '2.0',
        'id': req_id,
        'method': 'tasks/get',
        'params': params,
    }


def cancel_task_request(task_id: str, req_id: str = 'req-1') -> CancelTaskRequest:
    """Build a CancelTaskRequest."""
    return {
        'jsonrpc': '2.0',
        'id': req_id,
        'method': 'tasks/cancel',
        'params': {'id': task_id},
    }


def create_test_message(
    role: Literal['user', 'agent'] = 'user',
    text: str = 'Hello',
    message_id: str = 'msg-1',
    context_id: str | None = None,
) -> Message:
    """Helper to create a properly typed Message."""
    text_part: TextPart = {
        'kind': 'text',
        'text': text,
    }
    message: Message = {
        'role': role,  # No cast needed now!
        'parts': [text_part],
        'message_id': message_id,
        'kind': 'message',
    }
    if context_id is not None:
        message['context_id'] = context_id
    return message


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.asyncio
async def test_task_manager_context_manager(task_manager_factory: TaskManager):
    """Test TaskManager as async context manager."""
    # Not running before entering context
    assert not task_manager_factory.is_running

    async with task_manager_factory:
        # Should be running inside context
        assert task_manager_factory.is_running

    # Not running after exiting context
    assert not task_manager_factory.is_running


@pytest.mark.asyncio
async def test_task_manager_exit_without_enter(task_manager_factory: TaskManager):
    """Test exiting TaskManager without entering raises error."""
    with pytest.raises(RuntimeError, match='TaskManager was not properly initialized'):
        await task_manager_factory.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_send_message(
    task_manager: tuple[TaskManager, InMemoryBroker, InMemoryStorage], mocker: MockerFixture
) -> None:
    """Test send_message method."""
    tm, broker, _storage = task_manager

    # Mock broker.run_task
    mock_run_task = mocker.patch.object(broker, 'run_task', new_callable=AsyncMock)

    # Create and send message with metadata
    message = create_test_message(text='Hello', message_id='msg-1')
    metadata = {'user_id': '123'}
    request = send_message_request(message, req_id='req-1', metadata=metadata)

    response = await tm.send_message(request)

    # Verify response
    assert response['jsonrpc'] == '2.0'
    assert response['id'] == 'req-1'
    assert 'result' in response

    task = response['result']
    assert task['kind'] == 'task'
    assert task['status']['state'] == 'submitted'
    assert 'id' in task
    assert 'context_id' in task

    # Verify broker.run_task was called with metadata
    mock_run_task.assert_called_once()
    call_args = mock_run_task.call_args[0][0]
    assert call_args.get('metadata') == metadata


@pytest.mark.asyncio
async def test_send_message_with_context_id(
    task_manager: tuple[TaskManager, InMemoryBroker, InMemoryStorage], mocker: MockerFixture
) -> None:
    """Test send_message with explicit context_id."""
    tm, broker, _storage = task_manager

    # Mock broker.run_task
    mocker.patch.object(broker, 'run_task', new_callable=AsyncMock)

    # Create message with context_id
    message = create_test_message(text='Hello', message_id='msg-2', context_id='custom-context-123')
    request = send_message_request(message, req_id='req-2')

    response = await tm.send_message(request)

    assert 'result' in response
    task = response['result']
    assert 'context_id' in task
    assert task['context_id'] == 'custom-context-123'


@pytest.mark.asyncio
async def test_send_message_with_history_length(
    task_manager: tuple[TaskManager, InMemoryBroker, InMemoryStorage], mocker: MockerFixture
) -> None:
    """Test send_message with history_length configuration."""
    tm, broker, _storage = task_manager

    # Mock broker.run_task
    mock_run_task = mocker.patch.object(broker, 'run_task', new_callable=AsyncMock)

    # Create message with configuration
    message = create_test_message(text='Hello', message_id='msg-3')
    configuration: MessageSendConfiguration = {
        'history_length': 10,
        'accepted_output_modes': ['text/plain'],
    }
    request = send_message_request(message, req_id='req-3', configuration=configuration)

    await tm.send_message(request)

    # Verify history_length was passed to broker
    mock_run_task.assert_called_once()
    call_args = mock_run_task.call_args[0][0]
    assert 'history_length' in call_args
    assert call_args['history_length'] == 10


@pytest.mark.asyncio
async def test_get_task(
    task_manager: tuple[TaskManager, InMemoryBroker, InMemoryStorage], mocker: MockerFixture
) -> None:
    """Test get_task method."""
    tm, broker, _storage = task_manager

    # Mock broker.run_task
    mocker.patch.object(broker, 'run_task', new_callable=AsyncMock)

    # First create a task
    message = create_test_message(text='Hello', message_id='msg-4')
    send_request = send_message_request(message, req_id='req-4')

    send_response = await tm.send_message(send_request)
    assert 'result' in send_response
    assert 'id' in send_response['result']
    task_id = send_response['result']['id']

    # Now get the task
    get_request = get_task_request(task_id, req_id='req-5')
    get_response = await tm.get_task(get_request)

    assert get_response['jsonrpc'] == '2.0'
    assert get_response['id'] == 'req-5'
    assert 'result' in get_response
    assert get_response['result']['id'] == task_id


@pytest.mark.asyncio
async def test_get_task_not_found(task_manager: tuple[TaskManager, InMemoryBroker, InMemoryStorage]) -> None:
    """Test get_task with non-existent task."""
    tm, _broker, _storage = task_manager

    get_request = get_task_request('non-existent-task', req_id='req-6')
    get_response = await tm.get_task(get_request)

    assert get_response['jsonrpc'] == '2.0'
    assert get_response['id'] == 'req-6'
    assert 'error' in get_response
    assert get_response['error']['code'] == -32001
    assert get_response['error']['message'] == 'Task not found'


@pytest.mark.asyncio
async def test_get_task_with_history_length(
    task_manager: tuple[TaskManager, InMemoryBroker, InMemoryStorage], mocker: MockerFixture
) -> None:
    """Test get_task with history_length parameter."""
    tm, broker, storage = task_manager

    # Mock broker.run_task
    mocker.patch.object(broker, 'run_task', new_callable=AsyncMock)

    # Mock storage.load_task to track calls
    original_load_task = storage.load_task
    mock_load_task = mocker.patch.object(storage, 'load_task', new_callable=AsyncMock)
    # Make it return a valid task by calling the original
    mock_load_task.side_effect = original_load_task

    # Create a task first
    message = create_test_message(text='Hello', message_id='msg-7')
    send_request = send_message_request(message, req_id='req-7')

    send_response = await tm.send_message(send_request)
    assert 'result' in send_response
    assert 'id' in send_response['result']
    task_id = send_response['result']['id']

    # Get task with history_length
    get_request = get_task_request(task_id, req_id='req-8', history_length=5)
    await tm.get_task(get_request)

    # Verify history_length was passed
    mock_load_task.assert_called_with(task_id, 5)


@pytest.mark.asyncio
async def test_cancel_task(
    task_manager: tuple[TaskManager, InMemoryBroker, InMemoryStorage], mocker: MockerFixture
) -> None:
    """Test cancel_task method."""
    tm, broker, _storage = task_manager

    # Mock broker methods
    mocker.patch.object(broker, 'run_task', new_callable=AsyncMock)
    mock_cancel_task = mocker.patch.object(broker, 'cancel_task', new_callable=AsyncMock)

    # Create a task first
    message = create_test_message(text='Hello', message_id='msg-9')
    send_request = send_message_request(message, req_id='req-9')

    send_response = await tm.send_message(send_request)
    assert 'result' in send_response
    assert 'id' in send_response['result']
    task_id = send_response['result']['id']

    # Cancel the task
    cancel_request = cancel_task_request(task_id, req_id='req-10')
    cancel_response = await tm.cancel_task(cancel_request)

    assert cancel_response['jsonrpc'] == '2.0'
    assert cancel_response['id'] == 'req-10'
    assert 'result' in cancel_response
    assert cancel_response['result']['id'] == task_id

    # Verify broker.cancel_task was called
    mock_cancel_task.assert_called_once()
    assert mock_cancel_task.call_args[0][0]['id'] == task_id


@pytest.mark.asyncio
async def test_cancel_task_not_found(
    task_manager: tuple[TaskManager, InMemoryBroker, InMemoryStorage], mocker: MockerFixture
) -> None:
    """Test cancel_task with non-existent task."""
    tm, broker, _storage = task_manager

    # Mock broker.cancel_task
    mocker.patch.object(broker, 'cancel_task', new_callable=AsyncMock)

    cancel_request = cancel_task_request('non-existent-task', req_id='req-11')
    cancel_response = await tm.cancel_task(cancel_request)

    assert cancel_response['jsonrpc'] == '2.0'
    assert cancel_response['id'] == 'req-11'
    assert 'error' in cancel_response
    assert cancel_response['error']['code'] == -32001
    assert cancel_response['error']['message'] == 'Task not found'


@pytest.mark.asyncio
async def test_stream_message(
    task_manager: tuple[TaskManager, InMemoryBroker, InMemoryStorage], mocker: MockerFixture
) -> None:
    """Test stream_message method."""
    tm, broker, _storage = task_manager

    # Track the task info for our simulated worker
    task_info: dict[str, str | None] = {'task_id': None, 'context_id': None}

    # Create side effect for simulating worker
    async def simulate_worker(params: TaskSendParams):
        # Capture task info
        task_info['task_id'] = params['id']
        task_info['context_id'] = params['context_id']

        # Simulate worker behavior in background
        async def worker_task():
            # Small delay to ensure subscriber is ready
            await asyncio.sleep(0.1)

            # Send working status
            if task_info['task_id'] is not None and task_info['context_id'] is not None:
                status: TaskStatus = {'state': 'working'}
                event: TaskStatusUpdateEvent = {
                    'kind': 'status-update',
                    'task_id': task_info['task_id'],
                    'context_id': task_info['context_id'],
                    'status': status,
                    'final': False,
                }
                await broker.send_stream_event(task_info['task_id'], event)

            # Small delay for processing
            await asyncio.sleep(0.1)

            # Send completion status
            if task_info['task_id'] is not None and task_info['context_id'] is not None:
                status: TaskStatus = {'state': 'completed'}
                event: TaskStatusUpdateEvent = {
                    'kind': 'status-update',
                    'task_id': task_info['task_id'],
                    'context_id': task_info['context_id'],
                    'status': status,
                    'final': True,
                }
                await broker.send_stream_event(task_info['task_id'], event)

        # Start worker simulation in background
        asyncio.create_task(worker_task())

    # Mock broker.run_task with side effect
    mock_run_task = mocker.patch.object(broker, 'run_task', new_callable=AsyncMock, side_effect=simulate_worker)

    # Create stream request with metadata
    message = create_test_message(text='Hello streaming', message_id='msg-12')
    metadata = {'session_id': 'stream-123'}
    request = stream_message_request(message, req_id='req-12', metadata=metadata)

    # Collect events
    events: list[Any] = []
    async for event in tm.stream_message(request):
        events.append(event)

    # Should have received the task and status updates
    assert len(events) == 3  # task, working status, completed status
    assert events[0]['kind'] == 'task'
    assert events[1]['kind'] == 'status-update'
    assert events[1]['status']['state'] == 'working'
    assert events[2]['kind'] == 'status-update'
    assert events[2]['status']['state'] == 'completed'
    assert events[2]['final'] is True

    # Verify metadata was passed to broker
    mock_run_task.assert_called_once()
    call_args = mock_run_task.call_args[0][0]
    assert call_args.get('metadata') == metadata


@pytest.mark.asyncio
async def test_stream_message_with_context_and_history(
    task_manager: tuple[TaskManager, InMemoryBroker, InMemoryStorage], mocker: MockerFixture
) -> None:
    """Test stream_message with context_id and history_length."""
    tm, broker, _storage = task_manager

    # Create side effect for quick completion
    async def simulate_quick_completion(params: TaskSendParams):
        # Simulate worker behavior in background
        async def worker_task():
            # Small delay to ensure subscriber is ready
            await asyncio.sleep(0.1)

            # Send completion status immediately for this test
            status: TaskStatus = {'state': 'completed'}
            event: TaskStatusUpdateEvent = {
                'kind': 'status-update',
                'task_id': params['id'],
                'context_id': params['context_id'],
                'status': status,
                'final': True,
            }
            await broker.send_stream_event(params['id'], event)

        # Start worker simulation in background
        asyncio.create_task(worker_task())

    # Mock broker.run_task with side effect
    mock_run_task = mocker.patch.object(
        broker, 'run_task', new_callable=AsyncMock, side_effect=simulate_quick_completion
    )

    # Create message with context and configuration
    message = create_test_message(text='Hello', message_id='msg-13', context_id='stream-context-123')
    configuration: MessageSendConfiguration = {
        'history_length': 15,
        'accepted_output_modes': ['text/plain'],
    }
    request = stream_message_request(message, req_id='req-13', configuration=configuration)

    # Collect all events
    events: list[Any] = []
    async for event in tm.stream_message(request):
        events.append(event)

    # Verify we got the task and completion status
    assert len(events) == 2
    assert events[0]['kind'] == 'task'
    assert events[0]['context_id'] == 'stream-context-123'
    assert events[1]['kind'] == 'status-update'
    assert events[1]['final'] is True

    # Verify params passed to broker
    mock_run_task.assert_called_once()
    call_args = mock_run_task.call_args[0][0]
    assert call_args['context_id'] == 'stream-context-123'
    assert 'history_length' in call_args
    assert call_args['history_length'] == 15
