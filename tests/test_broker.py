"""Tests for the broker pub/sub functionality."""

import asyncio

import anyio
import pytest

from fasta2a.broker import InMemoryBroker, StreamEvent
from fasta2a.schema import Task, TaskArtifactUpdateEvent, TaskStatusUpdateEvent


@pytest.mark.asyncio
async def test_broker_pub_sub_single_subscriber():
    """Test basic pub/sub with a single subscriber."""
    async with InMemoryBroker() as broker:
        task_id = 'test-task-123'
        events_received = []

        # Create a task to track completion
        subscriber_done = asyncio.Event()

        async def subscriber():
            async for event in broker.subscribe_to_stream(task_id):
                events_received.append(event)
                # Check for final event
                if isinstance(event, dict) and event.get('kind') == 'status-update' and event.get('final'):
                    break
            subscriber_done.set()

        # Start subscriber in background
        subscriber_task = asyncio.create_task(subscriber())

        # Give subscriber time to set up
        await asyncio.sleep(0.1)

        # Send some events
        test_task: Task = {
            'id': task_id,
            'context_id': 'test-context',
            'kind': 'task',
            'status': {'state': 'submitted'},
        }
        await broker.send_stream_event(task_id, test_task)

        status_update: TaskStatusUpdateEvent = {
            'kind': 'status-update',
            'task_id': task_id,
            'context_id': 'test-context',
            'status': {'state': 'working'},
            'final': False,
        }
        await broker.send_stream_event(task_id, status_update)

        final_update: TaskStatusUpdateEvent = {
            'kind': 'status-update',
            'task_id': task_id,
            'context_id': 'test-context',
            'status': {'state': 'completed'},
            'final': True,
        }
        await broker.send_stream_event(task_id, final_update)

        # Wait for subscriber to complete
        await subscriber_done.wait()
        await subscriber_task

        # Verify events were received
        assert len(events_received) == 3
        assert events_received[0] == test_task
        assert events_received[1] == status_update
        assert events_received[2] == final_update


@pytest.mark.asyncio
async def test_broker_pub_sub_multiple_subscribers():
    """Test pub/sub with multiple subscribers to the same task."""
    async with InMemoryBroker() as broker:
        task_id = 'test-task-456'
        events_sub1 = []
        events_sub2 = []

        # Create completion events
        sub1_done = asyncio.Event()
        sub2_done = asyncio.Event()

        async def subscriber1():
            async for event in broker.subscribe_to_stream(task_id):
                events_sub1.append(event)
                if isinstance(event, dict) and event.get('kind') == 'status-update' and event.get('final'):
                    break
            sub1_done.set()

        async def subscriber2():
            async for event in broker.subscribe_to_stream(task_id):
                events_sub2.append(event)
                if isinstance(event, dict) and event.get('kind') == 'status-update' and event.get('final'):
                    break
            sub2_done.set()

        # Start both subscribers
        sub1_task = asyncio.create_task(subscriber1())
        sub2_task = asyncio.create_task(subscriber2())

        # Give subscribers time to set up
        await asyncio.sleep(0.1)

        # Send event
        test_event: TaskStatusUpdateEvent = {
            'kind': 'status-update',
            'task_id': task_id,
            'context_id': 'test-context',
            'status': {'state': 'completed'},
            'final': True,
        }
        await broker.send_stream_event(task_id, test_event)

        # Wait for both subscribers
        await sub1_done.wait()
        await sub2_done.wait()
        await sub1_task
        await sub2_task

        # Both should receive the event
        assert len(events_sub1) == 1
        assert len(events_sub2) == 1
        assert events_sub1[0] == test_event
        assert events_sub2[0] == test_event


@pytest.mark.asyncio
async def test_broker_no_subscribers():
    """Test sending events when there are no subscribers."""
    async with InMemoryBroker() as broker:
        task_id = 'test-task-789'

        # This should not raise an error even with no subscribers
        test_event: TaskStatusUpdateEvent = {
            'kind': 'status-update',
            'task_id': task_id,
            'context_id': 'test-context',
            'status': {'state': 'working'},
            'final': False,
        }
        await broker.send_stream_event(task_id, test_event)

        # Verify no subscribers exist
        assert task_id not in broker._event_subscribers


@pytest.mark.asyncio
async def test_broker_subscriber_cleanup_on_disconnect():
    """Test that disconnected subscribers are automatically cleaned up."""
    async with InMemoryBroker() as broker:
        task_id = 'test-task-cleanup'

        # Create a subscriber that exits early
        async def early_exit_subscriber():
            async for event in broker.subscribe_to_stream(task_id):
                # Exit after first event
                break

        # Start subscriber
        subscriber_task = asyncio.create_task(early_exit_subscriber())

        # Give subscriber time to set up
        await asyncio.sleep(0.1)

        # Verify subscriber is registered
        assert task_id in broker._event_subscribers
        assert len(broker._event_subscribers[task_id]) == 1

        # Send first event
        await broker.send_stream_event(
            task_id,
            {
                'kind': 'status-update',
                'task_id': task_id,
                'context_id': 'ctx',
                'status': {'state': 'working'},
                'final': False,
            },
        )

        # Wait for subscriber to exit
        await subscriber_task

        # Give time for cleanup to happen
        await asyncio.sleep(0.1)

        # Add a second subscriber to verify cleanup happens during send
        events_received = []
        complete = asyncio.Event()

        async def second_subscriber():
            async for event in broker.subscribe_to_stream(task_id):
                events_received.append(event)
                if isinstance(event, dict) and event.get('final'):
                    break
            complete.set()

        sub2_task = asyncio.create_task(second_subscriber())
        await asyncio.sleep(0.1)

        # Now send another event - the first (disconnected) subscriber should be cleaned up
        await broker.send_stream_event(
            task_id,
            {
                'kind': 'status-update',
                'task_id': task_id,
                'context_id': 'ctx',
                'status': {'state': 'completed'},
                'final': True,
            },
        )

        # Wait for second subscriber
        await complete.wait()
        await sub2_task

        # Verify the second subscriber got the event
        assert len(events_received) == 1
        assert events_received[0]['status']['state'] == 'completed'


@pytest.mark.asyncio
async def test_broker_artifact_streaming():
    """Test streaming artifact update events."""
    async with InMemoryBroker() as broker:
        task_id = 'test-task-artifacts'
        events_received = []

        complete = asyncio.Event()

        async def subscriber():
            async for event in broker.subscribe_to_stream(task_id):
                events_received.append(event)
                if isinstance(event, dict) and event.get('kind') == 'artifact-update' and event.get('last_chunk'):
                    break
            complete.set()

        # Start subscriber
        subscriber_task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.1)

        # Send artifact updates
        artifact1: TaskArtifactUpdateEvent = {
            'kind': 'artifact-update',
            'task_id': task_id,
            'context_id': 'test-context',
            'artifact': {'artifact_id': 'artifact-1', 'parts': [{'kind': 'text', 'text': 'Hello'}]},
            'append': False,
        }
        await broker.send_stream_event(task_id, artifact1)

        artifact2: TaskArtifactUpdateEvent = {
            'kind': 'artifact-update',
            'task_id': task_id,
            'context_id': 'test-context',
            'artifact': {'artifact_id': 'artifact-1', 'parts': [{'kind': 'text', 'text': ' World'}]},
            'append': True,
            'last_chunk': True,
        }
        await broker.send_stream_event(task_id, artifact2)

        # Wait for completion
        await complete.wait()
        await subscriber_task

        # Verify both artifacts received
        assert len(events_received) == 2
        assert events_received[0]['artifact']['parts'][0]['text'] == 'Hello'
        assert events_received[1]['artifact']['parts'][0]['text'] == ' World'
        assert events_received[1]['append'] is True
        assert events_received[1]['last_chunk'] is True


@pytest.mark.asyncio
async def test_broker_concurrent_operations():
    """Test concurrent pub/sub operations."""
    async with InMemoryBroker() as broker:
        num_tasks = 5
        num_events_per_task = 10

        results = {f'task-{i}': [] for i in range(num_tasks)}

        async def subscriber(task_id: str):
            async for event in broker.subscribe_to_stream(task_id):
                results[task_id].append(event)
                if isinstance(event, dict) and event.get('final'):
                    break

        async def publisher(task_id: str):
            for i in range(num_events_per_task - 1):
                await broker.send_stream_event(
                    task_id,
                    {
                        'kind': 'status-update',
                        'task_id': task_id,
                        'context_id': 'ctx',
                        'status': {'state': 'working'},
                        'message_num': i,
                        'final': False,
                    },
                )
                await asyncio.sleep(0.01)  # Small delay to test ordering

            # Send final event
            await broker.send_stream_event(
                task_id,
                {
                    'kind': 'status-update',
                    'task_id': task_id,
                    'context_id': 'ctx',
                    'status': {'state': 'completed'},
                    'message_num': num_events_per_task - 1,
                    'final': True,
                },
            )

        # Start all subscribers and publishers concurrently
        async with anyio.create_task_group() as tg:
            for i in range(num_tasks):
                task_id = f'task-{i}'
                tg.start_soon(subscriber, task_id)
                tg.start_soon(publisher, task_id)

        # Verify all events were received in order
        for i in range(num_tasks):
            task_id = f'task-{i}'
            assert len(results[task_id]) == num_events_per_task
            for j in range(num_events_per_task):
                assert results[task_id][j]['message_num'] == j


@pytest.mark.asyncio
async def test_broker_closed_stream_handling():
    """Test handling of closed streams when sending events."""
    async with InMemoryBroker() as broker:
        task_id = 'test-task-closed'

        # Create a stream and immediately close the receive side
        send_stream, receive_stream = anyio.create_memory_object_stream[StreamEvent](max_buffer_size=10)

        # Manually add the send stream to subscribers
        async with broker._subscriber_lock:
            broker._event_subscribers[task_id] = [send_stream]

        # Close the receive stream to simulate disconnection
        await receive_stream.aclose()

        # Now try to send an event - should handle the closed stream gracefully
        test_event: TaskStatusUpdateEvent = {
            'kind': 'status-update',
            'task_id': task_id,
            'context_id': 'test-context',
            'status': {'state': 'working'},
            'final': False,
        }

        # This should not raise an error
        await broker.send_stream_event(task_id, test_event)

        # Verify the closed stream was removed
        assert task_id not in broker._event_subscribers


@pytest.mark.asyncio
async def test_broker_early_final_event():
    """Test that subscription stops on receiving a final event."""
    async with InMemoryBroker() as broker:
        task_id = 'test-task-early-final'
        events_received = []

        async def subscriber():
            async for event in broker.subscribe_to_stream(task_id):
                events_received.append(event)

        # Start subscriber
        subscriber_task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.1)

        # Send a non-final event
        await broker.send_stream_event(
            task_id,
            {
                'kind': 'status-update',
                'task_id': task_id,
                'context_id': 'ctx',
                'status': {'state': 'working'},
                'final': False,
            },
        )

        # Send a final event - this should cause the subscriber to exit
        await broker.send_stream_event(
            task_id,
            {
                'kind': 'status-update',
                'task_id': task_id,
                'context_id': 'ctx',
                'status': {'state': 'completed'},
                'final': True,
            },
        )

        # Wait for subscriber to complete
        await subscriber_task

        # Send another event after subscriber has exited - no subscribers should receive this
        await broker.send_stream_event(
            task_id,
            {
                'kind': 'status-update',
                'task_id': task_id,
                'context_id': 'ctx',
                'status': {'state': 'working'},
                'final': False,
            },
        )

        # Should have received only 2 events (not the third)
        assert len(events_received) == 2
        assert events_received[1]['final'] is True


@pytest.mark.asyncio
async def test_broker_subscriber_double_removal():
    """Test edge case of trying to remove a subscriber twice."""
    async with InMemoryBroker() as broker:
        task_id = 'test-task-double-remove'

        # Create and manually manage a subscriber
        send_stream, receive_stream = anyio.create_memory_object_stream[StreamEvent](max_buffer_size=10)

        # Add subscriber
        async with broker._subscriber_lock:
            broker._event_subscribers[task_id] = [send_stream]

        # Remove it once
        async with broker._subscriber_lock:
            broker._event_subscribers[task_id].remove(send_stream)
            if not broker._event_subscribers[task_id]:
                del broker._event_subscribers[task_id]

        # Add it back to test the ValueError path
        async with broker._subscriber_lock:
            broker._event_subscribers[task_id] = [send_stream]

        # Now use subscribe_to_stream to create a proper subscription
        events = []

        async def test_subscriber():
            async for event in broker.subscribe_to_stream(task_id):
                events.append(event)
                # Exit after first event to trigger cleanup
                break

        # Start subscriber
        sub_task = asyncio.create_task(test_subscriber())
        await asyncio.sleep(0.1)

        # Manually remove the send_stream we added (not the one from subscribe_to_stream)
        async with broker._subscriber_lock:
            broker._event_subscribers[task_id].remove(send_stream)

        # Send event to trigger the subscriber to exit and cleanup
        await broker.send_stream_event(task_id, {'kind': 'test', 'data': 'test'})

        # Wait for subscriber
        await sub_task

        # Clean up
        await send_stream.aclose()
        await receive_stream.aclose()
