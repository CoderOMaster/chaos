# CLI reference

Installing the package provides a `chaos` console script (and the equivalent
`python -m chaos`).

## `chaos download`

Fetch and cache a model so an app's first run is instant and works offline.

```bash
chaos download                 # default: all-MiniLM-L6-v2
chaos download <model>         # a known short name, or any Hugging Face repo id
python -m chaos download       # equivalent, no console script needed
```

Prints the resolved local paths:

```
model: ~/.cache/huggingface/hub/models--Xenova--all-MiniLM-L6-v2/snapshots/…/onnx/model.onnx
vocab: ~/.cache/huggingface/hub/models--Xenova--all-MiniLM-L6-v2/snapshots/…/vocab.txt
```

Models are stored in the standard Hugging Face cache (`~/.cache/huggingface`, or
`$HF_HOME`). Once cached, `chaos download` and `Client()` use the files directly
with no network round-trip.

### When to use it

- **Docker / CI images** — run `chaos download` in the build so the image ships
  with the model and never downloads at runtime.
- **Offline / air-gapped** — pre-fetch on a connected machine, copy the HF cache
  across, then use normally (or pass `Client(model_path=…, vocab_path=…)`).
- **First-run latency** — avoid the one-time ~86 MB download on the first query.

Programmatic equivalent:

```python
from chaos import download_model
model_path, vocab_path = download_model("all-MiniLM-L6-v2")
```
