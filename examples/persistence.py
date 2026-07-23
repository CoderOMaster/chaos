"""Named indexes persist to ~/.chaos/<name>.jsonl and reload by name.
Also shows upsert: re-adding an existing id updates it.

    python examples/persistence.py
"""
from pathlib import Path

from chaos import Client, Document

NAME = "example_persist"
Path("~/.chaos").expanduser().joinpath(f"{NAME}.jsonl").unlink(missing_ok=True)  # clean start

# First client: create and populate the index.
c1 = Client()
idx = c1.open(NAME)
added, updated = idx.add([
    Document(id="1", text="Book flights for the offsite."),
    Document(id="2", text="Order the new laptop stands."),
])
print(f"first open : {added} added, {updated} updated, count={idx.count}")

# Upsert: same id, new text -> update, not add.
added, updated = idx.add([Document(id="2", text="Cancel the laptop stands; use monitor arms.")])
print(f"upsert id=2: {added} added, {updated} updated")

# A brand-new client re-opens the same name and reloads from disk (re-embeds).
c2 = Client()
idx2 = c2.open(NAME)
print(f"reopened   : count={idx2.count} (reloaded from disk)")
top = idx2.search("office equipment order", top_k=1)[0]
print(f"  top: {top.id} :: {top.text}")
