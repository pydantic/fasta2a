"""Tests for TaskManager class."""

import pytest

from fasta2a.broker import InMemoryBroker
from fasta2a.schema import TaskIdParams, TaskSendParams
from fasta2a.storage import InMemoryStorage
from fasta2a.task_manager import TaskManager


@pytest.mark.asyncio
async def test_task_manager_context_manager():
    """Test TaskManager as async context manager."""
    storage = InMemoryStorage()
    broker = InMemoryBroker()
    task_manager = TaskManager(broker=broker, storage=storage)

    # Not running before entering context
    assert not task_manager.is_running

    async with task_manager:
        # Should be running inside context
        assert task_manager.is_running

    # Not running after exiting context
    assert not task_manager.is_running


@pytest.mark.asyncio
async def test_task_manager_exit_without_enter():
    """Test exiting TaskManager without entering raises error."""
    storage = InMemoryStorage()
    broker = InMemoryBroker()
    task_manager = TaskManager(broker=broker, storage=storage)

    with pytest.raises(RuntimeError, match='TaskManager was not properly initialized'):
        await task_manager.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_send_message():
    """Test send_message method."""
    storage = InMemoryStorage()
    broker = InMemoryBroker()
    task_manager = TaskManager(broker=broker, storage=storage)

    async with broker:
        async with task_manager:
            request = {
                'jsonrpc': '2.0',
                'id': 'req-1',
                'method': 'message/send',
                'params': {
                    'message': {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': 'Hello'}],
                        'messageId': 'msg-1',
                        'kind': 'message',
                    }
                },
            }

            response = await task_manager.send_message(request)

            assert response['jsonrpc'] == '2.0'
            assert response['id'] == 'req-1'
            assert 'result' in response

            task = response['result']
            assert task['kind'] == 'task'
            assert task['status']['state'] == 'submitted'
            assert 'id' in task
            assert 'contextId' in task


@pytest.mark.asyncio
async def test_send_message_with_context_id():
    """Test send_message with explicit context_id."""
    storage = InMemoryStorage()
    broker = InMemoryBroker()
    task_manager = TaskManager(broker=broker, storage=storage)

    async with broker:
        async with task_manager:
            request = {
                'jsonrpc': '2.0',
                'id': 'req-2',
                'method': 'message/send',
                'params': {
                    'message': {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': 'Hello'}],
                        'messageId': 'msg-2',
                        'kind': 'message',
                        'context_id': 'custom-context-123',
                    }
                },
            }

            response = await task_manager.send_message(request)
            task = response['result']
            assert task['contextId'] == 'custom-context-123'


@pytest.mark.asyncio
async def test_send_message_with_history_length():
    """Test send_message with history_length configuration."""
    storage = InMemoryStorage()
    broker = InMemoryBroker()
    task_manager = TaskManager(broker=broker, storage=storage)

    # Mock broker.run_task to capture params
    run_task_calls = []
    original_run_task = broker.run_task

    async def mock_run_task(params: TaskSendParams):
        run_task_calls.append(params)
        return await original_run_task(params)

    broker.run_task = mock_run_task

    async with broker:
        async with task_manager:
            request = {
                'jsonrpc': '2.0',
                'id': 'req-3',
                'method': 'message/send',
                'params': {
                    'message': {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': 'Hello'}],
                        'messageId': 'msg-3',
                        'kind': 'message',
                    },
                    'configuration': {'history_length': 10},
                },
            }

            await task_manager.send_message(request)

            # Verify history_length was passed to broker
            assert len(run_task_calls) == 1
            assert run_task_calls[0]['history_length'] == 10


@pytest.mark.asyncio
async def test_get_task():
    """Test get_task method."""
    storage = InMemoryStorage()
    broker = InMemoryBroker()
    task_manager = TaskManager(broker=broker, storage=storage)

    async with broker:
        async with task_manager:
            # First create a task
            send_request = {
                'jsonrpc': '2.0',
                'id': 'req-4',
                'method': 'message/send',
                'params': {
                    'message': {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': 'Hello'}],
                        'messageId': 'msg-4',
                        'kind': 'message',
                    }
                },
            }

            send_response = await task_manager.send_message(send_request)
            task_id = send_response['result']['id']

            # Now get the task
            get_request = {'jsonrpc': '2.0', 'id': 'req-5', 'method': 'tasks/get', 'params': {'id': task_id}}

            get_response = await task_manager.get_task(get_request)

            assert get_response['jsonrpc'] == '2.0'
            assert get_response['id'] == 'req-5'
            assert 'result' in get_response
            assert get_response['result']['id'] == task_id


@pytest.mark.asyncio
async def test_get_task_not_found():
    """Test get_task with non-existent task."""
    storage = InMemoryStorage()
    broker = InMemoryBroker()
    task_manager = TaskManager(broker=broker, storage=storage)

    async with broker:
        async with task_manager:
            get_request = {
                'jsonrpc': '2.0',
                'id': 'req-6',
                'method': 'tasks/get',
                'params': {'id': 'non-existent-task'},
            }

            get_response = await task_manager.get_task(get_request)

            assert get_response['jsonrpc'] == '2.0'
            assert get_response['id'] == 'req-6'
            assert 'error' in get_response
            assert get_response['error']['code'] == -32001
            assert get_response['error']['message'] == 'Task not found'


@pytest.mark.asyncio
async def test_get_task_with_history_length():
    """Test get_task with history_length parameter."""
    storage = InMemoryStorage()
    broker = InMemoryBroker()
    task_manager = TaskManager(broker=broker, storage=storage)

    # Mock storage.load_task to capture history_length
    load_task_calls = []
    original_load_task = storage.load_task

    async def mock_load_task(task_id: str, history_length: int | None = None):
        load_task_calls.append((task_id, history_length))
        return await original_load_task(task_id, history_length)

    storage.load_task = mock_load_task

    async with broker:
        async with task_manager:
            # Create a task first
            send_request = {
                'jsonrpc': '2.0',
                'id': 'req-7',
                'method': 'message/send',
                'params': {
                    'message': {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': 'Hello'}],
                        'messageId': 'msg-7',
                        'kind': 'message',
                    }
                },
            }

            send_response = await task_manager.send_message(send_request)
            task_id = send_response['result']['id']

            # Get task with history_length
            get_request = {
                'jsonrpc': '2.0',
                'id': 'req-8',
                'method': 'tasks/get',
                'params': {'id': task_id, 'history_length': 5},
            }

            await task_manager.get_task(get_request)

            # Verify history_length was passed
            assert any(call[0] == task_id and call[1] == 5 for call in load_task_calls)


@pytest.mark.asyncio
async def test_cancel_task():
    """Test cancel_task method."""
    storage = InMemoryStorage()
    broker = InMemoryBroker()
    task_manager = TaskManager(broker=broker, storage=storage)

    # Track cancel calls
    cancel_calls = []
    original_cancel = broker.cancel_task

    async def mock_cancel(params: TaskIdParams):
        cancel_calls.append(params)
        return await original_cancel(params)

    broker.cancel_task = mock_cancel

    async with broker:
        async with task_manager:
            # Create a task first
            send_request = {
                'jsonrpc': '2.0',
                'id': 'req-9',
                'method': 'message/send',
                'params': {
                    'message': {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': 'Hello'}],
                        'messageId': 'msg-9',
                        'kind': 'message',
                    }
                },
            }

            send_response = await task_manager.send_message(send_request)
            task_id = send_response['result']['id']

            # Cancel the task
            cancel_request = {'jsonrpc': '2.0', 'id': 'req-10', 'method': 'tasks/cancel', 'params': {'id': task_id}}

            cancel_response = await task_manager.cancel_task(cancel_request)

            assert cancel_response['jsonrpc'] == '2.0'
            assert cancel_response['id'] == 'req-10'
            assert 'result' in cancel_response
            assert cancel_response['result']['id'] == task_id

            # Verify broker.cancel_task was called
            assert len(cancel_calls) == 1
            assert cancel_calls[0]['id'] == task_id


@pytest.mark.asyncio
async def test_cancel_task_not_found():
    """Test cancel_task with non-existent task."""
    storage = InMemoryStorage()
    broker = InMemoryBroker()
    task_manager = TaskManager(broker=broker, storage=storage)

    async with broker:
        async with task_manager:
            cancel_request = {
                'jsonrpc': '2.0',
                'id': 'req-11',
                'method': 'tasks/cancel',
                'params': {'id': 'non-existent-task'},
            }

            cancel_response = await task_manager.cancel_task(cancel_request)

            assert cancel_response['jsonrpc'] == '2.0'
            assert cancel_response['id'] == 'req-11'
            assert 'error' in cancel_response
            assert cancel_response['error']['code'] == -32001
            assert cancel_response['error']['message'] == 'Task not found'


@pytest.mark.asyncio
async def test_stream_message():
    """Test stream_message method."""
    storage = InMemoryStorage()
    broker = InMemoryBroker()
    task_manager = TaskManager(broker=broker, storage=storage)

    async with broker:
        async with task_manager:
            request = {
                'jsonrpc': '2.0',
                'id': 'req-12',
                'method': 'message/stream',
                'params': {
                    'message': {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': 'Hello streaming'}],
                        'messageId': 'msg-12',
                        'kind': 'message',
                    }
                },
            }

            events = []
            async for event in task_manager.stream_message(request):
                events.append(event)
                # Simulate sending a final event to end the stream
                if len(events) == 1 and event['kind'] == 'task':
                    task_id = event['id']
                    await broker.send_stream_event(
                        task_id,
                        {
                            'kind': 'status-update',
                            'task_id': task_id,
                            'context_id': event['contextId'],
                            'status': {'state': 'completed'},
                            'final': True,
                        },
                    )

            # Should have received at least the task and final status
            assert len(events) >= 2
            assert events[0]['kind'] == 'task'
            assert events[-1]['kind'] == 'status-update'
            assert events[-1]['final'] is True


@pytest.mark.asyncio
async def test_stream_message_with_context_and_history():
    """Test stream_message with context_id and history_length."""
    storage = InMemoryStorage()
    broker = InMemoryBroker()
    task_manager = TaskManager(broker=broker, storage=storage)

    # Track run_task calls
    run_task_calls = []

    async def mock_run(params: TaskSendParams):
        run_task_calls.append(params)
        # Don't actually run to avoid background tasks
        pass

    broker.run_task = mock_run

    async with broker:
        async with task_manager:
            request = {
                'jsonrpc': '2.0',
                'id': 'req-13',
                'method': 'message/stream',
                'params': {
                    'message': {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': 'Hello'}],
                        'messageId': 'msg-13',
                        'kind': 'message',
                        'context_id': 'stream-context-123',
                    },
                    'configuration': {'history_length': 15},
                },
            }

            # Get first event (the task)
            async for event in task_manager.stream_message(request):
                if event['kind'] == 'task':
                    task_id = event['id']
                    # End stream immediately
                    await broker.send_stream_event(
                        task_id,
                        {
                            'kind': 'status-update',
                            'task_id': task_id,
                            'context_id': 'stream-context-123',
                            'status': {'state': 'completed'},
                            'final': True,
                        },
                    )

            # Verify params passed to broker
            assert len(run_task_calls) == 1
            assert run_task_calls[0]['context_id'] == 'stream-context-123'
            assert run_task_calls[0]['history_length'] == 15
