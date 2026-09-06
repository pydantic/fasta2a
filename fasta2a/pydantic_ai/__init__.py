"""Pydantic AI bridge for FastA2A.

Port of `pydantic_ai._a2a` into the FastA2A package: Pydantic AI 2 removed
`Agent.to_a2a()`, and `agent_to_a2a` here is its replacement. See
https://github.com/pydantic/pydantic-ai for the upstream context.

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
