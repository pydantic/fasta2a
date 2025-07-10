# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastA2A is a Python library built on top of Starlette and Pydantic that provides an easy way to convert AI agents into A2A (Agent-to-Agent) servers. It implements the A2A protocol for agent communication.

## Development Commands

- `scripts/install` - Install dependencies using uv
- `scripts/test` - Run tests with coverage (`uv run coverage run -m pytest`)
- `scripts/lint` - Format and fix code (`uvx ruff check --fix && uvx ruff format`)
- `scripts/check` - Run lint checks, type checking, and validation (`uvx ruff format --check --diff`, `uvx ruff check`, `uv run pyright`, `uvx check-sdist`, `uv lock`)

## Architecture

The library follows a modular architecture with these key components:

### Core Components

- **FastA2A**: Main Starlette application class that handles HTTP endpoints and agent card configuration
- **TaskManager**: Orchestrates task execution between broker and storage
- **Broker**: Abstract interface for task scheduling (supports both in-memory and remote worker implementations)
- **Storage**: Abstract interface for persisting tasks and messages
- **Worker**: Processes tasks received from the broker
- **Schema**: Comprehensive A2A protocol schema definitions using Pydantic

### Key Endpoints

- `/.well-known/agent.json` - Agent card endpoint (GET/HEAD/OPTIONS)
- `/` - Main A2A protocol endpoint (POST) for message/send, tasks/get, tasks/cancel operations

### Protocol Implementation

The library implements the A2A protocol v0.2.5, supporting:
- Task-based message exchange
- JSON-RPC 2.0 communication
- Agent capabilities and skill definitions
- Task state management (submitted, working, completed, failed, etc.)

## Dependencies

Built with modern Python tooling:
- **uv** for dependency management
- **Starlette** for ASGI web framework
- **Pydantic** for data validation and serialization
- **OpenTelemetry** for observability
- **Ruff** for linting and formatting
- **Pyright** for type checking
- **pytest** for testing