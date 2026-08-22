"""Two embedders, same interface.

HashEmbedder is the keyless baseline: a deterministic bag-of-words vector
(feature hashing + L2 norm). It is deliberately simple - the point of the repo
is the retrieval GATE, and a baseline that can be beaten gives the gate
something to say. NVIDIAEmbedder calls nv-embed-v1 (the model this pipeline
was originally built against for the NVIDIA DLI assessment) via the NIM API.
"""
import hashlib
import json
import math
import os
import re
import urllib.request

DIM = 512
# Without this, function words dominate the cosine and every query drifts
# toward the longest document. The first baseline shipped without it - and
# failed the gate at recall 0.83. That failure is preserved in the README.
STOPWORDS = frozenset("""a an and are as at be by can do does for from has have how
in is it its of on or that the this to under up what when where which why will with""".split())


class HashEmbedder:
    name = "hash-bow-512-nostop"

    def embed(self, text):
        vec = [0.0] * DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            if tok in STOPWORDS:
                continue
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


class NVIDIAEmbedder:
    """nv-embed-v1 via https://integrate.api.nvidia.com (OpenAI-compatible).
    Needs NVIDIA_API_KEY (nvapi-...) in the environment. stdlib-only on purpose."""
    name = "nvidia/nv-embed-v1"

    def embed(self, text, input_type="passage"):
        req = urllib.request.Request(
            "https://integrate.api.nvidia.com/v1/embeddings",
            data=json.dumps({"model": "nvidia/nv-embed-v1", "input": [text[:2000]],
                             "input_type": input_type, "truncate": "END"}).encode(),
            headers={"Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["data"][0]["embedding"]


def get_embedder(kind):
    return NVIDIAEmbedder() if kind == "nvidia" else HashEmbedder()
