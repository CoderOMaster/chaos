"""Metadata round-trips with matches (it is returned, not searched). Use it to
filter or group results on the client side.

    python examples/metadata.py
"""
from chaos import Client, Document

client = Client()
index = client.open()

index.add([
    Document(id="1", text="Fix the flaky checkout test.",      metadata={"project": "web",    "priority": "high"}),
    Document(id="2", text="Tune the retriever latency budget.", metadata={"project": "search", "priority": "high"}),
    Document(id="3", text="Write the onboarding doc.",          metadata={"project": "web",    "priority": "low"}),
])

# Search semantically, then filter by metadata in Python.
matches = index.search("engineering work to do", top_k=10)
print("high-priority matches:")
for m in matches:
    if m.metadata.get("priority") == "high":
        print(f"  [{m.metadata['project']}] {m.score:.3f}  {m.text}")
