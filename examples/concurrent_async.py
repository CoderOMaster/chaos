"""Async: run many searches concurrently. Because the native engine releases the
GIL, `asyncio.gather` overlaps the searches across worker threads rather than
serializing them.

    python examples/concurrent_async.py
"""
import asyncio
import time

from chaos import AsyncClient, Document


async def main():
    client = AsyncClient()
    index = await client.open()
    await index.add([
        Document(id="1", text="The cat napped on the warm windowsill."),
        Document(id="2", text="Rust has no garbage collector."),
        Document(id="3", text="Vector search returns nearest neighbors."),
        Document(id="4", text="The orchestra rehearsed the final movement."),
    ])

    queries = ["a sleeping pet", "language without a garbage collector",
               "finding nearest neighbors", "an orchestra performance",
               "warm afternoon sunlight", "memory-safe systems programming"]

    t0 = time.perf_counter()
    results = await asyncio.gather(*(index.search(q, top_k=1) for q in queries))
    dt = (time.perf_counter() - t0) * 1e3

    for q, hits in zip(queries, results):
        print(f"{q:<22} -> {hits[0].text}")
    print(f"\n{len(queries)} concurrent searches in {dt:.1f} ms")


asyncio.run(main())
