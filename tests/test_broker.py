"""Tests for the broker pub/sub functionality."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import anyio
import pytest

from fasta2a.broker import InMemoryBroker, StreamEvent
from fasta2a.schema import Task, TaskArtifactUpdateEvent, TaskState, TaskStatusUpdateEvent


def make_status_event(
    task_id: str, state: TaskState = 'working', final: bool = False, metadata: dict[str, Any] | None = None
) -> TaskStatusUpdateEvent:
    """Create a TaskStatusUpdateEvent with common defaults."""
    event: TaskStatusUpdateEvent = {
        'kind': 'status-update',
        'task_id': task_id,
        'context_id': 'test-context',
        'status': {'state': state},
        'final': final,
    }
    if metadata:
        event['metadata'] = metadata
    return event


def make_artifact_event(
    task_id: str, text: str, append: bool = False, last_chunk: bool = False
) -> TaskArtifactUpdateEvent:
    """Create a TaskArtifactUpdateEvent with common defaults."""
    return {
        'kind': 'artifact-update',
        'task_id': task_id,
        'context_id': 'test-context',
        'artifact': {'artifact_id': 'artifact-1', 'parts': [{'kind': 'text', 'text': text}]},
        'append': append,
        'last_chunk': last_chunk,
    }


def is_final_status_event(event: StreamEvent) -> bool:
    """Check if event is a final status update."""
    return isinstance(event, dict) and event.get('kind') == 'status-update' and bool(event.get('final'))


@pytest.mark.asyncio
async def test_broker_pub_sub_single_subscriber():
    """Test basic pub/sub with a single subscriber."""
    async with InMemoryBroker() as broker:
        task_id = 'test-task-123'
        events_received: list[StreamEvent] = []

        # Create events to track subscriber lifecycle
        subscriber_ready = asyncio.Event()
        subscriber_done = asyncio.Event()

        async def subscriber():
            # Set ready event to signal we're about to start listening
            subscriber_ready.set()
            async for event in broker.subscribe_to_stream(task_id):
                events_received.append(event)
                # Check for final event
                if is_final_status_event(event):
                    break
            subscriber_done.set()

        # Start subscriber in background
        subscriber_task = asyncio.create_task(subscriber())

        # Wait for subscriber to be ready
        await subscriber_ready.wait()

        # Send some events
        test_task: Task = {
            'id': task_id,
            'context_id': 'test-context',
            'kind': 'task',
            'status': {'state': 'submitted'},
        }
        await broker.send_stream_event(task_id, test_task)

        status_update = make_status_event(task_id, 'working')
        await broker.send_stream_event(task_id, status_update)

        final_update = make_status_event(task_id, 'completed', final=True)
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
        events_sub1: list[StreamEvent] = []
        events_sub2: list[StreamEvent] = []

        # Create ready and completion events
        sub1_ready = asyncio.Event()
        sub2_ready = asyncio.Event()
        sub1_done = asyncio.Event()
        sub2_done = asyncio.Event()

        async def subscriber1():
            sub1_ready.set()
            async for event in broker.subscribe_to_stream(task_id):
                events_sub1.append(event)
                if is_final_status_event(event):
                    break
            sub1_done.set()

        async def subscriber2():
            sub2_ready.set()
            async for event in broker.subscribe_to_stream(task_id):
                events_sub2.append(event)
                if is_final_status_event(event):
                    break
            sub2_done.set()

        # Start both subscribers
        sub1_task = asyncio.create_task(subscriber1())
        sub2_task = asyncio.create_task(subscriber2())

        # Wait for both subscribers to be ready
        await sub1_ready.wait()
        await sub2_ready.wait()

        # Send event
        test_event = make_status_event(task_id, 'completed', final=True)
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
        test_event = make_status_event(task_id, 'working')
        await broker.send_stream_event(task_id, test_event)

        # Test passes if no error is raised when sending with no subscribers


@pytest.mark.asyncio
async def test_broker_subscriber_cleanup_on_disconnect():
    """Test that broker continues to work correctly after subscribers disconnect."""
    async with InMemoryBroker() as broker:
        task_id = 'test-task-cleanup'
        first_subscriber_events: list[StreamEvent] = []

        # Create ready event for first subscriber
        first_subscriber_ready = asyncio.Event()

        # Create a subscriber that exits early
        async def early_exit_subscriber():
            first_subscriber_ready.set()
            async for event in broker.subscribe_to_stream(task_id):
                first_subscriber_events.append(event)
                # Exit after first event
                break

        # Start subscriber
        subscriber_task = asyncio.create_task(early_exit_subscriber())

        # Wait for subscriber to be ready
        await first_subscriber_ready.wait()

        # Send first event
        await broker.send_stream_event(task_id, make_status_event(task_id, 'working'))

        # Wait for subscriber to exit
        await subscriber_task

        # Verify first subscriber received the event before disconnecting
        assert len(first_subscriber_events) == 1

        # Add a second subscriber to verify system continues working
        events_received: list[StreamEvent] = []
        second_subscriber_ready = asyncio.Event()
        complete = asyncio.Event()

        async def second_subscriber():
            second_subscriber_ready.set()
            async for event in broker.subscribe_to_stream(task_id):
                events_received.append(event)
                if isinstance(event, dict) and event.get('final'):
                    break
            complete.set()

        sub2_task = asyncio.create_task(second_subscriber())
        await second_subscriber_ready.wait()

        # Send another event - verifies broker works after first subscriber disconnected
        await broker.send_stream_event(task_id, make_status_event(task_id, 'completed', final=True))

        # Wait for second subscriber
        await complete.wait()
        await sub2_task

        # Verify the second subscriber got only the new event (not the old one)
        assert len(events_received) == 1
        event = events_received[0]
        assert isinstance(event, dict) and event.get('kind') == 'status-update'
        # Type narrow to TaskStatusUpdateEvent
        status_event = cast(TaskStatusUpdateEvent, event)
        assert status_event['status']['state'] == 'completed'


@pytest.mark.asyncio
async def test_broker_artifact_streaming():
    """Test streaming artifact update events."""
    async with InMemoryBroker() as broker:
        task_id = 'test-task-artifacts'
        events_received: list[StreamEvent] = []

        subscriber_ready = asyncio.Event()
        complete = asyncio.Event()

        async def subscriber():
            subscriber_ready.set()
            async for event in broker.subscribe_to_stream(task_id):
                events_received.append(event)
                if isinstance(event, dict) and event.get('kind') == 'artifact-update' and event.get('last_chunk'):
                    break
            complete.set()

        # Start subscriber
        subscriber_task = asyncio.create_task(subscriber())
        await subscriber_ready.wait()

        # Send artifact updates
        artifact1 = make_artifact_event(task_id, 'Hello')
        await broker.send_stream_event(task_id, artifact1)

        artifact2 = make_artifact_event(task_id, ' World', append=True, last_chunk=True)
        await broker.send_stream_event(task_id, artifact2)

        # Wait for completion
        await complete.wait()
        await subscriber_task

        # Verify both artifacts received
        assert len(events_received) == 2

        # Check first artifact
        event0 = events_received[0]
        assert isinstance(event0, dict) and event0.get('kind') == 'artifact-update'
        # Type narrow to TaskArtifactUpdateEvent
        artifact_event0 = cast(TaskArtifactUpdateEvent, event0)
        artifact_parts = artifact_event0['artifact']['parts']
        assert len(artifact_parts) > 0
        part = artifact_parts[0]
        assert part['kind'] == 'text' and 'text' in part and part['text'] == 'Hello'

        # Check second artifact
        event1 = events_received[1]
        assert isinstance(event1, dict) and event1.get('kind') == 'artifact-update'
        # Type narrow to TaskArtifactUpdateEvent
        artifact_event1 = cast(TaskArtifactUpdateEvent, event1)
        artifact_parts = artifact_event1['artifact']['parts']
        assert len(artifact_parts) > 0
        part = artifact_parts[0]
        assert part['kind'] == 'text' and 'text' in part and part['text'] == ' World'
        assert artifact_event1.get('append') is True
        assert artifact_event1.get('last_chunk') is True


@pytest.mark.asyncio
async def test_broker_concurrent_operations():
    """Test concurrent pub/sub operations."""
    async with InMemoryBroker() as broker:
        num_tasks = 5
        num_events_per_task = 10

        results: dict[str, list[StreamEvent]] = {f'task-{i}': [] for i in range(num_tasks)}

        async def subscriber(task_id: str):
            async for event in broker.subscribe_to_stream(task_id):
                results[task_id].append(event)
                if isinstance(event, dict) and event.get('final'):
                    break

        async def publisher(task_id: str):
            for i in range(num_events_per_task - 1):
                event = make_status_event(task_id, 'working', metadata={'message_num': i})
                await broker.send_stream_event(task_id, event)

            # Send final event
            final_event = make_status_event(
                task_id, 'completed', final=True, metadata={'message_num': num_events_per_task - 1}
            )
            await broker.send_stream_event(task_id, final_event)

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
                event = results[task_id][j]
                assert isinstance(event, dict) and 'metadata' in event
                assert event['metadata']['message_num'] == j


@pytest.mark.asyncio
async def test_broker_early_final_event():
    """Test that subscription stops on receiving a final event."""
    async with InMemoryBroker() as broker:
        task_id = 'test-task-early-final'
        events_received: list[StreamEvent] = []

        subscriber_ready = asyncio.Event()

        async def subscriber():
            subscriber_ready.set()
            async for event in broker.subscribe_to_stream(task_id):
                events_received.append(event)

        # Start subscriber
        subscriber_task = asyncio.create_task(subscriber())
        await subscriber_ready.wait()

        # Send a non-final event
        await broker.send_stream_event(task_id, make_status_event(task_id, 'working'))

        # Send a final event - this should cause the subscriber to exit
        await broker.send_stream_event(task_id, make_status_event(task_id, 'completed', final=True))

        # Wait for subscriber to complete
        await subscriber_task

        # Send another event after subscriber has exited - no subscribers should receive this
        await broker.send_stream_event(task_id, make_status_event(task_id, 'working'))

        # Should have received only 2 events (not the third)
        assert len(events_received) == 2
        # Check the second event is final
        event = events_received[1]
        assert isinstance(event, dict) and event.get('kind') == 'status-update'
        status_event = cast(TaskStatusUpdateEvent, event)
        assert status_event['final'] is True
