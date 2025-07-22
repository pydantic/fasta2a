from __future__ import annotations as _annotations

from contextlib import asynccontextmanager

import httpx
import pytest
from asgi_lifespan import LifespanManager
from inline_snapshot import snapshot

from fasta2a.applications import FastA2A
from fasta2a.broker import InMemoryBroker
from fasta2a.storage import InMemoryStorage

pytestmark = pytest.mark.anyio


@asynccontextmanager
async def create_test_client(app: FastA2A):
    async with LifespanManager(app=app) as manager:
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport, base_url='http://testclient') as client:
            yield client


async def test_agent_card():
    app = FastA2A(storage=InMemoryStorage(), broker=InMemoryBroker())
    async with create_test_client(app) as client:
        response = await client.get('/.well-known/agent.json')
        assert response.status_code == 200
        assert response.json() == snapshot(
            {
                'name': 'My Agent',
                'description': 'An AI agent exposed as an A2A agent.',
                'url': 'http://localhost:8000',
                'version': '1.0.0',
                'protocolVersion': '0.2.5',
                'skills': [],
                'defaultInputModes': ['application/json'],
                'defaultOutputModes': ['application/json'],
                'capabilities': {
                    'streaming': False,
                    'pushNotifications': False,
                    'stateTransitionHistory': False,
                },
            }
        )


async def test_agent_card_with_all_params():
    """Test agent card with all parameters specified."""
    app = FastA2A(
        storage=InMemoryStorage(),
        broker=InMemoryBroker(),
        name='Test Agent',
        url='https://example.com',
        version='2.0.0',
        description='A test agent',
        provider='Test Provider',
        skills=['skill1', 'skill2'],
        streaming=True,
    )
    async with create_test_client(app) as client:
        response = await client.get('/.well-known/agent.json')
        assert response.status_code == 200
        data = response.json()
        assert data['name'] == 'Test Agent'
        assert data['url'] == 'https://example.com'
        assert data['version'] == '2.0.0'
        assert data['description'] == 'A test agent'
        assert data['provider'] == 'Test Provider'
        assert data['skills'] == ['skill1', 'skill2']
        assert data['capabilities']['streaming'] is True


async def test_agent_card_head_and_options():
    """Test HEAD and OPTIONS methods for agent card."""
    app = FastA2A(storage=InMemoryStorage(), broker=InMemoryBroker())
    async with create_test_client(app) as client:
        # Test HEAD
        head_response = await client.head('/.well-known/agent.json')
        assert head_response.status_code == 200

        # Test OPTIONS
        options_response = await client.options('/.well-known/agent.json')
        assert options_response.status_code == 200


async def test_agent_card_caching():
    """Test that agent card is cached after first generation."""
    app = FastA2A(storage=InMemoryStorage(), broker=InMemoryBroker(), name='Original')
    async with create_test_client(app) as client:
        # First request
        response1 = await client.get('/.well-known/agent.json')
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1['name'] == 'Original'

        # Modify app (shouldn't affect cached response)
        app.name = 'Modified'

        # Second request should return cached
        response2 = await client.get('/.well-known/agent.json')
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2['name'] == 'Original'


async def test_docs_endpoint():
    """Test the /docs endpoint."""
    app = FastA2A(storage=InMemoryStorage(), broker=InMemoryBroker())
    async with create_test_client(app) as client:
        response = await client.get('/docs')
        assert response.status_code == 200
        assert response.headers['content-type'] == 'text/html; charset=utf-8'
        assert b'<!DOCTYPE html>' in response.content or b'<html' in response.content
