"""Synchronous basics — add documents to an in-memory index and search.

    python examples/sync_basic.py
"""
from chaos import Client, Document

client = Client()              # MiniLM downloaded + cached on first use
index = client.open()          # no name -> in-memory; kind="flat" (exact) by default

index.add([
    Document(id="a", text="The espresso machine needs descaling this week."),
    Document(id="b", text="Renew the domain name before it expires."),
    Document(id="c", text="Draft the Q3 board update and share it."),
    Document(id="d", text="Grind fresh beans for tomorrow's pour-over."),
])
print(f"indexed {index.count} docs\n")

for query in ["coffee maintenance", "financial reporting"]:
    print(f"query: {query!r}")
    for m in index.search(query, top_k=2):
        print(f"  {m.score:.3f}  {m.text}")
    print()
