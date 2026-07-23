"""chaos quickstart.

    python examples/quickstart.py

No model path needed — MiniLM is downloaded and cached on first run.
(Pre-fetch with `python -m chaos download` if you prefer.)
"""
import asyncio

from chaos import AsyncClient, Document


async def main():
    client = AsyncClient()                       # auto-resolves MiniLM
    index = await client.open("notes")           # kind="flat" (default) or "hnsw"

    added, updated = await index.add([
        Document(id="1", text="Ship the on-device SDK by Friday."),
        Document(id="2", text="Follow up with the LiveKit team about latency."),
        Document(id="3", text="Buy oat milk and coffee beans on the way home."),
    ])
    print(f"{added} added, {updated} updated, {index.count} total")

    for m in await index.search("what's due this week", top_k=3):
        print(f"{m.id} score={m.score:.3f} {m.text}")


asyncio.run(main())
