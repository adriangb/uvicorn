from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator

import pytest

from uvicorn._types import ASGIReceiveCallable, ASGISendCallable, Scope
from uvicorn.config import Config
from uvicorn.protocols.http import connection_limits
from uvicorn.protocols.http.connection_limits import MAX_AGE_JITTER, ConnectionLimits
from uvicorn.protocols.http.h11_impl import H11Protocol
from uvicorn.server import Server

pytestmark = pytest.mark.anyio

HTTPProtocol = type[H11Protocol]


class App:
    """`/slow` waits for `release` before it responds; every other path responds at once."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.started = asyncio.Event()

    async def __call__(self, scope: Scope, receive: ASGIReceiveCallable, send: ASGISendCallable) -> None:
        assert scope["type"] == "http"
        if scope["path"] == "/slow":
            self.started.set()
            await self.release.wait()
        body = b"Hello, world"
        headers = [(b"content-type", b"text/plain"), (b"content-length", str(len(body)).encode())]
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        await send({"type": "http.response.body", "body": body})


class Client:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = writer

    def send_request(self, path: str = "/") -> None:
        self.writer.write(f"GET {path} HTTP/1.1\r\nHost: example.org\r\n\r\n".encode())

    async def read_response(self) -> tuple[bytes, dict[bytes, bytes], bytes]:
        head = await asyncio.wait_for(self.reader.readuntil(b"\r\n\r\n"), 5)
        status, *lines = head.split(b"\r\n")[:-2]
        headers = dict((name.lower(), value.strip()) for name, value in (line.split(b":", 1) for line in lines))
        body = await asyncio.wait_for(self.reader.readexactly(int(headers[b"content-length"])), 5)
        return status, headers, body

    async def request(self, path: str = "/") -> tuple[bytes, dict[bytes, bytes], bytes]:
        self.send_request(path)
        return await self.read_response()

    async def assert_closed(self) -> None:
        """The server closed the connection cleanly (FIN, not RST), with nothing more to read."""
        assert await asyncio.wait_for(self.reader.read(), 5) == b""

    async def assert_open(self) -> None:
        """The server has neither closed the connection nor sent anything unprompted."""
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(self.reader.read(1), 0.2)


@contextlib.asynccontextmanager
async def serve(app: App, port: int, http_protocol_cls: HTTPProtocol, **kwargs: object) -> AsyncIterator[Server]:
    config = Config(app=app, port=port, http=http_protocol_cls, lifespan="off", loop="asyncio", **kwargs)  # type: ignore[arg-type]
    server = Server(config=config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    try:
        yield server
    finally:
        server.should_exit = True
        await task


@contextlib.asynccontextmanager
async def connect(port: int) -> AsyncIterator[Client]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        yield Client(reader, writer)
    finally:
        writer.close()
        with contextlib.suppress(ConnectionError):
            await writer.wait_closed()


async def test_connection_max_requests(http_protocol_cls: HTTPProtocol, unused_tcp_port: int):
    app = App()
    async with serve(app, unused_tcp_port, http_protocol_cls, limit_connection_max_requests=3):
        async with connect(unused_tcp_port) as client:
            for _ in range(2):
                status, headers, _ = await client.request()
                assert status == b"HTTP/1.1 200 OK"
                assert b"connection" not in headers
            status, headers, body = await client.request()
            assert status == b"HTTP/1.1 200 OK"
            assert headers[b"connection"] == b"close"
            assert body == b"Hello, world"
            await client.assert_closed()

        # The limit is per connection: a new connection starts from zero.
        async with connect(unused_tcp_port) as client:
            _, headers, _ = await client.request()
            assert b"connection" not in headers


async def test_connection_max_age(http_protocol_cls: HTTPProtocol, unused_tcp_port: int):
    app = App()
    async with serve(app, unused_tcp_port, http_protocol_cls, limit_connection_max_age=0.5):
        async with connect(unused_tcp_port) as client:
            _, headers, _ = await client.request()
            assert b"connection" not in headers

            # Past its maximum age, the idle connection is not closed: it serves its next request, then closes.
            await asyncio.sleep(0.6)
            await client.assert_open()
            status, headers, _ = await client.request()
            assert status == b"HTTP/1.1 200 OK"
            assert headers[b"connection"] == b"close"
            await client.assert_closed()


async def test_connection_max_age_does_not_cut_in_flight_request(http_protocol_cls: HTTPProtocol, unused_tcp_port: int):
    app = App()
    async with serve(app, unused_tcp_port, http_protocol_cls, limit_connection_max_age=0.2):
        async with connect(unused_tcp_port) as client:
            client.send_request("/slow")
            await app.started.wait()
            await asyncio.sleep(0.4)
            await client.assert_open()
            app.release.set()

            status, headers, body = await client.read_response()
            assert status == b"HTTP/1.1 200 OK"
            assert headers[b"connection"] == b"close"
            assert body == b"Hello, world"
            await client.assert_closed()


async def test_no_connection_limits_by_default(http_protocol_cls: HTTPProtocol, unused_tcp_port: int):
    app = App()
    async with serve(app, unused_tcp_port, http_protocol_cls):
        async with connect(unused_tcp_port) as client:
            for _ in range(5):
                _, headers, _ = await client.request()
                assert b"connection" not in headers
            await client.assert_open()


def test_connection_max_age_jitter_is_bounded(monkeypatch: pytest.MonkeyPatch):
    config = Config(app=App(), limit_connection_max_age=100)
    for _ in range(1000):
        before = time.monotonic()
        limits = ConnectionLimits(config)
        after = time.monotonic()
        assert limits.deadline is not None
        assert before + 100 * (1 - MAX_AGE_JITTER) <= limits.deadline <= after + 100

    # The extremes of the jitter.
    for jitter, max_age in [(0.0, 100.0), (MAX_AGE_JITTER, 90.0)]:
        monkeypatch.setattr(connection_limits.random, "uniform", lambda a, b, jitter=jitter: jitter)
        monkeypatch.setattr(connection_limits.time, "monotonic", lambda: 1000.0)
        assert ConnectionLimits(config).deadline == pytest.approx(1000 + max_age)


def test_connection_limits_unset():
    limits = ConnectionLimits(Config(app=App()))
    assert limits.deadline is None
    assert not any(limits.response_started() for _ in range(100))
