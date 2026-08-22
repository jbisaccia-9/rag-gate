"""Agentic retrieval: a tool-calling loop that decomposes compound questions.

Single-shot RAG embeds the whole question once; on a compound question ("X and
Y") one topic's chunks crowd out the other's. The agent is handed a `search`
tool and may call it as many times as it needs - the scripted policy splits
the question into sub-queries and searches each, which is query decomposition,
the simplest genuinely agentic retrieval pattern.

Backends share one loop:
  * NVIDIAChatModel - tool-calling chat completions on the NVIDIA NIM API
    (NVIDIA_API_KEY; model via NVIDIA_CHAT_MODEL). The live path.
  * ScriptedModel  - deterministic decomposition policy for keyless CI. Its
    answer is composed from the REAL chunks the loop retrieved.

The answer faces a GROUNDING GATE: every [doc#chunk] citation must be a chunk
the agent actually retrieved this run. An answer citing chunks it never saw is
refused - retrieval theater is the failure mode this gate exists for.
"""
import json
import os
import re
import urllib.request

SYSTEM = ("Answer the user's question using the search tool. Decompose compound "
          "questions and search each part. Cite every claim as [doc#chunk]. "
          "Reply with the final answer only when done.")


def make_tools(index):
    retrieved = []

    def search(query, k=2):
        hits = index.search_ids(query, k=int(k))
        retrieved.extend(cid for _, cid, _ in hits)
        return [{"doc": d, "chunk": cid, "text": t[:200]} for d, cid, t in hits]

    schema = [{"type": "function", "function": {
        "name": "search", "description": "semantic search over the document index",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"},
                                      "k": {"type": "integer"}},
                       "required": ["query"]}}}]
    return {"search": search}, schema, retrieved


class NVIDIAChatModel:
    name = "nvidia-chat"

    def complete(self, messages, tools):
        req = urllib.request.Request(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            data=json.dumps({"model": os.environ.get("NVIDIA_CHAT_MODEL",
                                                     "meta/llama-3.3-70b-instruct"),
                             "messages": messages, "tools": tools}).encode(),
            headers={"Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["choices"][0]["message"]


class ScriptedModel:
    """Decomposition policy: split the question on its conjunction, search each
    sub-query, then answer from the retrieved chunks with citations."""
    name = "scripted"

    def __init__(self):
        self.subqueries = None
        self.hits = []

    def complete(self, messages, tools):
        if self.subqueries is None:
            question = messages[1]["content"]
            parts = re.split(r"\band\b", question, maxsplit=1)
            self.subqueries = [p.strip(" ?.,") for p in parts if p.strip()]
        last = messages[-1]
        if last.get("role") == "tool":
            self.hits.extend(json.loads(last["content"]))
        if self.subqueries:
            q = self.subqueries.pop(0)
            return {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"s{len(self.hits)}", "type": "function",
                 "function": {"name": "search",
                              "arguments": json.dumps({"query": q, "k": 2})}}]}
        lines = ["Answer assembled from retrieved passages:"]
        for h in self.hits:
            lines.append(f"- [{h['chunk']}] {h['text'][:90]}...")
        return {"role": "assistant", "content": "\n".join(lines)}


class HallucinatingModel(ScriptedModel):
    """Cites a chunk it never retrieved - the grounding gate's reason to exist."""
    name = "hallucinating"

    def complete(self, messages, tools):
        msg = super().complete(messages, tools)
        if msg.get("content"):
            msg["content"] += "\n- [ghost-doc#9] a passage the agent never saw"
        return msg


def run_agent(model, index, question, max_turns=8):
    registry, schema, retrieved = make_tools(index)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    for _ in range(max_turns):
        msg = model.complete(messages, schema)
        messages.append(msg)
        calls = msg.get("tool_calls")
        if not calls:
            return msg.get("content", ""), retrieved
        for call in calls:
            args = json.loads(call["function"]["arguments"] or "{}")
            result = registry[call["function"]["name"]](**args)
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps(result)})
    raise RuntimeError("agent exceeded max turns")


def grounding_gate(answer, retrieved):
    cited = set(re.findall(r"\[([\w-]+#\d+)\]", answer))
    ghosts = sorted(cited - set(retrieved))
    print(f"  {'PASS' if not ghosts else 'FAIL'}  answer grounding: "
          f"{len(cited)} citations, {len(ghosts)} never retrieved")
    for g in ghosts:
        print(f"    UNGROUNDED: [{g}]")
    if ghosts:
        print("ANSWER GATE: FAILED - the answer cites chunks the agent never saw.")
        return 1
    print("ANSWER GATE: PASSED - every citation was actually retrieved.")
    return 0
