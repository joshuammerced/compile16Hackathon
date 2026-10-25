"""Shared plumbing for the model services: a bounded single-worker queue and /health, /ready.

Each model service is one process holding one model. Requests are handled on the event loop,
but the model call runs on a dedicated worker thread (ThreadPoolExecutor with `workers`
threads, 1 for MLX models), so only one generation touches the GPU at a time and the server
keeps answering /ready while busy. When more than `max_queue` requests are waiting, new ones
get 503 + Retry-After instead of piling up forever.
"""

import asyncio
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# voice_pipeline.py and inference.py live one folder up and use flat imports.
LLM_DIR = Path(__file__).resolve().parents[1]
if str(LLM_DIR) not in sys.path:
    sys.path.insert(0, str(LLM_DIR))

log = logging.getLogger("services")


class Busy(Exception):
    """Queue full; the client should retry later."""


class WorkTimeout(Exception):
    """The request waited or ran longer than the service timeout."""


class ModelWorker:
    def __init__(self, name: str, workers: int = 1, max_queue: int = 8, timeout_s: float = 120.0):
        self.name = name
        self.workers = workers
        self.max_queue = max_queue
        self.timeout_s = timeout_s
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=name)
        self._lock = threading.Lock()
        self._pending = 0  # waiting + running
        self._running = 0
        self.served = 0

    @property
    def queued(self) -> int:
        return self._pending - self._running

    def stats(self) -> dict:
        return {"running": self._running, "queued": self.queued, "max_queue": self.max_queue, "served": self.served}

    def _call(self, fn, args):
        with self._lock:
            self._running += 1
        try:
            return fn(*args)
        finally:
            with self._lock:
                self._running -= 1
                self.served += 1

    def _done(self, _):
        with self._lock:
            self._pending -= 1

    async def run(self, fn, *args):
        """Run fn(*args) on the worker thread. Raises Busy or WorkTimeout."""
        with self._lock:
            if self._pending >= self.workers + self.max_queue:
                raise Busy(f"{self.name} queue full ({self.max_queue} waiting)")
            self._pending += 1
        future = self._pool.submit(self._call, fn, args)
        future.add_done_callback(self._done)
        try:
            # On timeout the job is dropped if it hasn't started yet.
            # A generation already running can't be interrupted; it finishes and is discarded.
            return await asyncio.wait_for(asyncio.wrap_future(future), self.timeout_s)
        except asyncio.TimeoutError:
            raise WorkTimeout(f"{self.name} took longer than {self.timeout_s:.0f}s") from None

    def shutdown(self):
        self._pool.shutdown(wait=False, cancel_futures=True)


def add_common_routes(app: FastAPI, name: str) -> None:
    """/health, /ready and the Busy/WorkTimeout -> 503/504 mapping.

    The lifespan must set app.state.worker (a ModelWorker) once the model is loaded.
    """
    started = time.time()

    @app.get("/health")
    async def health():
        return {"service": name, "status": "up", "uptime_s": round(time.time() - started, 1)}

    @app.get("/ready")
    async def ready():
        worker = getattr(app.state, "worker", None)
        if worker is None:
            return JSONResponse({"service": name, "ready": False}, status_code=503)
        return {"service": name, "ready": True, "model": app.state.model_name, **worker.stats()}

    @app.exception_handler(Busy)
    async def busy(_: Request, e: Busy):
        return JSONResponse({"error": str(e)}, status_code=503, headers={"Retry-After": "5"})

    @app.exception_handler(WorkTimeout)
    async def timeout(_: Request, e: WorkTimeout):
        return JSONResponse({"error": str(e)}, status_code=504)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
