"""A2A protocol extensions: what an agent supports, and what a request activated.

An extension is a capability negotiated by URI, on top of the core protocol
(https://a2a-protocol.org/latest/topics/extensions/). The agent lists the
ones it supports in its card; a client asks for some of them in the
``A2A-Extensions`` request header; the agent activates the ones it knows,
tells the client which in the same header on the response, and makes the list
available to whatever handles the task.

This module is the vocabulary for that exchange. `FastA2A` does the
negotiating; a `Worker` reads the outcome with `activated_extensions`.
"""

from __future__ import annotations as _annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

from .schema import AgentExtension

__all__ = (
    'A2A_EXTENSIONS_HEADER',
    'ACTIVATED_EXTENSIONS_KEY',
    'activated_extensions',
    'format_extensions_header',
    'missing_required_extensions',
    'parse_extensions_header',
    'select_activated_extensions',
)

A2A_EXTENSIONS_HEADER = 'A2A-Extensions'
"""The header a client activates extensions with, and the agent answers on."""

ACTIVATED_EXTENSIONS_KEY = 'a2a.activated_extensions'
"""Where the activated URIs travel in the request's ``metadata``, down to the worker."""


def parse_extensions_header(value: str | None) -> list[str]:
    """The extension URIs an ``A2A-Extensions`` header names.

    Comma-separated; whitespace around a URI is dropped, so is a repeated one,
    and the order the client wrote them in is kept.
    """
    if not value:
        return []
    uris: list[str] = []
    for raw in value.split(','):
        uri = raw.strip()
        if uri and uri not in uris:
            uris.append(uri)
    return uris


def format_extensions_header(uris: Iterable[str]) -> str:
    """The header value for a list of URIs."""
    return ', '.join(uris)


def select_activated_extensions(supported: Iterable[AgentExtension], requested: Iterable[str]) -> list[str]:
    """Of the URIs a client asked for, the ones this agent supports — in the client's order.

    A URI the agent never declared is not activated and not echoed back, which
    is how the client learns it was ignored.
    """
    known = {extension['uri'] for extension in supported}
    return [uri for uri in requested if uri in known]


def missing_required_extensions(supported: Iterable[AgentExtension], activated: Iterable[str]) -> list[str]:
    """The extensions this agent marks ``required`` that the client did not activate."""
    active = set(activated)
    return [extension['uri'] for extension in supported if extension.get('required') and extension['uri'] not in active]


def activated_extensions(params: Mapping[str, Any]) -> list[str]:
    """The extension URIs activated for a request or a task.

    Read from the ``metadata`` of `MessageSendParams` or `TaskSendParams`, where
    `FastA2A` records them — so a worker asks this of the params it was handed
    and gets what the client and the agent agreed on for that message.
    """
    metadata = params.get('metadata')
    if not isinstance(metadata, Mapping):
        return []
    value: object = cast('Mapping[str, object]', metadata).get(ACTIVATED_EXTENSIONS_KEY)
    if not isinstance(value, list):
        return []
    return [str(uri) for uri in cast('list[object]', value)]
