"""Shared fixtures: a disposable mock endpoint for every test that needs one."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mock_endpoint import MockServer  # path set up above


@pytest.fixture
def mock_factory():
    servers = []

    def make(**options):
        server = MockServer(**options)
        servers.append(server)
        return server

    yield make
    for server in servers:
        server.stop()


@pytest.fixture
def mock(mock_factory):
    return mock_factory()
