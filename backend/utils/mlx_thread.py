"""Single dedicated thread for all MLX operations.

MLX registers GPU streams per-thread: a model loaded on one thread cannot
be evaluated from another without raising "There is no Stream(gpu, N) in
current thread" (issue #675). Depending on the call path that either fails
the request or escapes as an uncaught C++ exception inside MLX's
StreamThread and aborts the whole server (SIGABRT in
``mlx::core::metal::get_command_encoder``).

``asyncio.to_thread`` hands work to the default executor pool, so load and
inference land on arbitrary, usually different, threads. Routing every MLX
load/eval call through this one long-lived worker guarantees the thread
that created the stream is the thread that uses it. MLX inference is
GPU-serial anyway, so a single worker costs no throughput.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-worker")


async def run_on_mlx_thread(fn, /, *args, **kwargs):
    """Run ``fn(*args, **kwargs)`` on the dedicated MLX worker thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, partial(fn, *args, **kwargs))
