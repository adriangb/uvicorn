from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from collections.abc import AsyncIterator, Generator

import pytest

from uvicorn._types import ASGIReceiveCallable, ASGISendCallable, Scope
from uvicorn.config import Config
from uvicorn.protocols.http.h11_impl import H11Protocol
from uvicorn.server import Server

pytestmark = pytest.mark.anyio

HTTPProtocol = type[H11Protocol]


class App:
    """`/slow` waits for `release` before it responds, `/split` sends its headers, then waits for `release`
    before it sends its body. Every other path responds at once."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.started = asyncio.Event()

    async def __call__(self, scope: Scope, receive: ASGIReceiveCallable, send: ASGISendCallable) -> None:
        assert scope["type"] == "http"
        body = b"Hello, world"
        headers = [(b"content-type", b"text/plain"), (b"content-length", str(len(body)).encode())]
        if scope["path"] == "/slow":
            self.started.set()
            await self.release.wait()
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        if scope["path"] == "/split":
            self.started.set()
            await self.release.wait()
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
async def serve(config: Config) -> AsyncIterator[tuple[Server, asyncio.Task[None]]]:
    server = Server(config=config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    try:
        yield server, task
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


@contextlib.contextmanager
def ignore_reraised_sigterm() -> Generator[None, None, None]:
    """`Server.serve()` raises the signals it handled once it returns: swallow that SIGTERM."""
    original_handler = signal.signal(signal.SIGTERM, lambda signum, frame: None)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, original_handler)


def make_config(app: App, port: int, http_protocol_cls: HTTPProtocol, **kwargs: object) -> Config:
    return Config(app=app, port=port, http=http_protocol_cls, lifespan="off", loop="asyncio", **kwargs)  # type: ignore[arg-type]


async def test_lame_duck_closes_connection_after_next_response(http_protocol_cls: HTTPProtocol, unused_tcp_port: int):
    app = App()
    async with serve(make_config(app, unused_tcp_port, http_protocol_cls)) as (server, _):
        async with connect(unused_tcp_port) as client:
            status, headers, _ = await client.request()
            assert status == b"HTTP/1.1 200 OK"
            assert b"connection" not in headers

            assert not server.lame_duck
            server.enter_lame_duck()
            assert server.lame_duck

            status, headers, body = await client.request()
            assert status == b"HTTP/1.1 200 OK"
            assert headers[b"connection"] == b"close"
            assert body == b"Hello, world"
            await client.assert_closed()


async def test_lame_duck_keeps_accepting_connections(http_protocol_cls: HTTPProtocol, unused_tcp_port: int):
    app = App()
    async with serve(make_config(app, unused_tcp_port, http_protocol_cls)) as (server, _):
        server.enter_lame_duck()
        await asyncio.sleep(0.2)  # Let the main loop tick: it must not stop the server.
        for _ in range(3):
            async with connect(unused_tcp_port) as client:
                status, headers, _ = await client.request()
                assert status == b"HTTP/1.1 200 OK"
                assert headers[b"connection"] == b"close"
                await client.assert_closed()


async def test_lame_duck_does_not_close_idle_connections(http_protocol_cls: HTTPProtocol, unused_tcp_port: int):
    app = App()
    async with serve(make_config(app, unused_tcp_port, http_protocol_cls)) as (server, _):
        async with connect(unused_tcp_port) as client:
            await client.request()
            server.enter_lame_duck()
            await asyncio.sleep(0.2)
            await client.assert_open()

            # The idle connection still serves its next request, then closes.
            status, headers, _ = await client.request()
            assert status == b"HTTP/1.1 200 OK"
            assert headers[b"connection"] == b"close"
            await client.assert_closed()


async def test_lame_duck_idle_connection_closes_on_keep_alive_timeout(
    http_protocol_cls: HTTPProtocol, unused_tcp_port: int
):
    app = App()
    config = make_config(app, unused_tcp_port, http_protocol_cls, timeout_keep_alive=1)
    async with serve(config) as (server, _):
        async with connect(unused_tcp_port) as client:
            await client.request()
            server.enter_lame_duck()
            await client.assert_closed()


async def test_lame_duck_does_not_cut_in_flight_request(http_protocol_cls: HTTPProtocol, unused_tcp_port: int):
    app = App()
    async with serve(make_config(app, unused_tcp_port, http_protocol_cls)) as (server, _):
        async with connect(unused_tcp_port) as client:
            client.send_request("/slow")
            await app.started.wait()
            server.enter_lame_duck()
            await asyncio.sleep(0.2)
            app.release.set()

            status, headers, body = await client.read_response()
            assert status == b"HTTP/1.1 200 OK"
            assert headers[b"connection"] == b"close"
            assert body == b"Hello, world"
            await client.assert_closed()


async def test_lame_duck_response_started_before_lame_duck(http_protocol_cls: HTTPProtocol, unused_tcp_port: int):
    """A response whose headers went out without `Connection: close` keeps the connection open: the next
    response carries the header and closes it."""
    app = App()
    async with serve(make_config(app, unused_tcp_port, http_protocol_cls)) as (server, _):
        async with connect(unused_tcp_port) as client:
            client.send_request("/split")
            await app.started.wait()
            server.enter_lame_duck()
            app.release.set()

            status, headers, body = await client.read_response()
            assert status == b"HTTP/1.1 200 OK"
            assert b"connection" not in headers
            assert body == b"Hello, world"
            await client.assert_open()

            status, headers, _ = await client.request()
            assert headers[b"connection"] == b"close"
            await client.assert_closed()


async def test_shutdown_after_lame_duck(http_protocol_cls: HTTPProtocol, unused_tcp_port: int):
    app = App()
    async with serve(make_config(app, unused_tcp_port, http_protocol_cls)) as (server, task):
        async with connect(unused_tcp_port) as client:
            await client.request()
            server.enter_lame_duck()
            server.should_exit = True
            # The idle connection is closed by the shutdown, as without lame duck mode.
            await client.assert_closed()
            await asyncio.wait_for(task, 5)
        with pytest.raises(OSError):
            await asyncio.open_connection("127.0.0.1", unused_tcp_port)


async def test_lame_duck_timeout(
    http_protocol_cls: HTTPProtocol, unused_tcp_port: int, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    app = App()
    async with serve(make_config(app, unused_tcp_port, http_protocol_cls)) as (server, task):
        server.enter_lame_duck(timeout=0.3)
        # A second call does not push the deadline back.
        server.enter_lame_duck(timeout=60)
        async with connect(unused_tcp_port) as client:
            _, headers, _ = await client.request()
            assert headers[b"connection"] == b"close"
        assert not task.done()
        await asyncio.wait_for(task, 5)
    assert caplog.text.count("Entering lame duck mode") == 1
    assert "Lame duck timeout elapsed." in caplog.text


async def test_sigterm_enters_lame_duck_when_timeout_lame_duck_is_set(
    http_protocol_cls: HTTPProtocol, unused_tcp_port: int, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    app = App()
    config = make_config(app, unused_tcp_port, http_protocol_cls, timeout_lame_duck=0.5)
    with ignore_reraised_sigterm():
        async with serve(config) as (server, task):
            server.handle_exit(signal.SIGTERM, None)
            assert not server.should_exit
            await asyncio.sleep(0.2)
            assert server.lame_duck
            async with connect(unused_tcp_port) as client:
                _, headers, _ = await client.request()
                assert headers[b"connection"] == b"close"
                await client.assert_closed()
            assert not task.done()
            await asyncio.wait_for(task, 5)
    assert "Lame duck timeout elapsed." in caplog.text


async def test_second_sigterm_skips_lame_duck_period(http_protocol_cls: HTTPProtocol, unused_tcp_port: int):
    app = App()
    config = make_config(app, unused_tcp_port, http_protocol_cls, timeout_lame_duck=60)
    with ignore_reraised_sigterm():
        async with serve(config) as (server, task):
            server.handle_exit(signal.SIGTERM, None)
            await asyncio.sleep(0.2)
            assert server.lame_duck
            server.handle_exit(signal.SIGTERM, None)
            assert server.should_exit
            await asyncio.wait_for(task, 5)


async def test_sigterm_without_timeout_lame_duck_shuts_down(http_protocol_cls: HTTPProtocol, unused_tcp_port: int):
    app = App()
    with ignore_reraised_sigterm():
        async with serve(make_config(app, unused_tcp_port, http_protocol_cls)) as (server, task):
            server.handle_exit(signal.SIGTERM, None)
            assert server.should_exit
            await asyncio.wait_for(task, 5)
            assert not server.lame_duck
