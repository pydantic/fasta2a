"""Pydantic AI bridge for FastA2A.

Port of `pydantic_ai._a2a` into the FastA2A package, so users can keep using
`Agent.to_a2a()` after the corresponding wrapper is removed from pydantic-ai
in v2. See https://github.com/pydantic/pydantic-ai for the upstream
deprecation context.

Usage:

    from pydantic_ai import Agent
    from fasta2a.pydantic_ai import agent_to_a2a

    agent = Agent('openai:gpt-5.5')
    app = agent_to_a2a(agent, name='my-agent', url='http://localhost:8000')

Install the integration with the `pydantic-ai` extra:

    pip install 'fasta2a[pydantic-ai]'
"""

from ._bridge import AgentWorker, agent_to_a2a, worker_lifespan

__all__ = ['AgentWorker', 'agent_to_a2a', 'worker_lifespan']
