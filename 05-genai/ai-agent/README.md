# Autonomous AI Agent

An agent that plans, calls tools, recovers from failures, and — the part that
actually matters — **always stops.**

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your-key-here

python -m agent tools                                    # what it can do
python -m agent run "summarise the notes" --workspace ~/notes
python -m agent replay traces/20260805T211535.jsonl
pytest -q                                                # 141 tests, no network
```

See [DEPLOY.md](./DEPLOY.md) for Docker and unattended runs.

## Termination is a property, not a hope

An agent whose stopping condition is "the model decides to stop" is an agent
that sometimes does not. Four independent limits are checked **before every
model call**:

| Limit | Why |
|---|---|
| Step budget | the obvious one |
| Token budget | a cheap loop is still an expensive loop |
| Consecutive failures | 3 failures in a row means stuck, not working |
| Guardrail halt | a hard stop wins over anything the model wants |

The consecutive-failure counter **resets on success**, deliberately. An agent
that fails, recovers, and fails again is working; one that fails three times
running is not. Three tests drive the agent into infinite loops and assert it
comes back.

```python
state = Agent(looping_client, config=AgentConfig(max_steps=5)).run("loop forever")
assert state.status is Status.BUDGET_EXCEEDED
assert state.step_count == 5
```

## Failures are observations, not exceptions

Every tool returns a `ToolResult` instead of raising. An exception escaping the
loop ends the run — exactly the wrong response to "file not found", which the
agent could recover from in one step.

```
[ERR] 1. read_file(path='/nope.txt') -> FileNotFoundError: no such file
[ok ] 2. list_files(directory='.')
[ok ] 3. finish(answer='...')
```

Hallucinated tool names get the same treatment, and the error names the real
tools so the model can correct itself:

```
no such tool 'summon_unicorn'. Available: calculate, read_file, list_files, ...
```

## Guardrails are controls, not prompts

Tools are capabilities and an LLM decides when to use them. That needs a layer
which does not consult the model, because the model is the thing being
constrained. **Prompting is a request; a guardrail is a control.**

**Path confinement** resolves symlinks and `..` *first*, then compares:

```python
workspace/../../../etc/passwd   # blocked — resolves outside
innocent.md -> /etc/shadow      # blocked — symlink followed first
```

String-prefix comparison is the classic bypass, since that path literally
starts with the workspace. Both cases have tests.

**Secrets are refused even inside the workspace** — `.env`, `*.pem`, `id_rsa`,
`*credentials*`.

**Destructive tools default to denied.** With writes enabled but no approver
configured, a destructive call is **denied, not allowed** — an unattended run
cannot ask anyone, so it does not get to write. Defaulting the other way is how
a guardrail becomes decoration.

## Being stuck is detected mechanically

Asking the model "are you stuck?" costs a round trip and gets an unreliable
answer. Comparing the last few calls costs nothing and is exact:

- **Identical repeat** — same tool, same arguments, twice
- **Repeated failure** — same tool failing, arguments varying
- **Thrashing** — alternating between two tools with no progress
- **Budget pressure** — 70% of steps spent, nothing finished

Each produces a nudge injected into the conversation, not a silent abort.

## Memory: compact before trimming

Tool results are the bulky part of a transcript — a `read_file` can be
thousands of tokens that matter for one turn. Two mechanisms, in this order:

1. **Compaction** summarises *old* tool results and keeps every step.
2. **Trimming** drops old messages only if compaction was not enough.

On a 10-step conversation that is 5,395 tokens → 1,901 compacted, with no step
lost. The task message is never dropped: losing it is the one failure an agent
cannot recover from.

## Three bugs the live run found

The offline suite passed 141 tests before any of these appeared. Running it for
real found all three:

**Gemini 3.x requires `thought_signature`.** Replaying a `functionCall` back
into history without the signature it arrived with is rejected outright:

```
400: Function call is missing a thought_signature in functionCall parts
```

It now survives the round trip.

**Relative paths resolved against the wrong directory.** The model asked for
`notes.md`, which resolved against the *process* working directory rather than
`--workspace`, and the guardrail blocked it — correctly, but for a reason that
looked arbitrary. `resolve_args` now rebases relative paths onto the workspace
before the check.

**The CLI could only read `.env` from the CWD.** Added `--env-file`.

Worth noting the guardrail did its job in the middle of that: it blocked a read
of a path outside the workspace that the agent genuinely asked for.

## Tests run offline

141 tests, no network, no API key. `ScriptedTransport` replays canned turns and
repeats the last one forever — which is what lets a test drive the agent into a
loop and prove the budget stops it.

One of those tests exists because of a bug in the test double itself: it stored
`payload` by reference, and `payload["contents"]` **is** the agent's live
message list, so recorded calls mutated retroactively. A real transport
serialises at call time; the fake now deep-copies.

## LangGraph is optional

`build_langgraph()` compiles the same three node functions into a `StateGraph`
and behaves identically. It is imported on demand — the graph is a scheduler,
and making it a hard dependency would mean the agent could not run without it.

## Layout

```
agent/
├── llm.py         Gemini client with tool calling; rate-limit handling
├── tools.py       registry; schemas generated from type hints
├── state.py       AgentState, Step, and the stopping conditions
├── graph.py       plan / act / observe; optional LangGraph
├── memory.py      compaction, trimming, scratchpad
├── reflect.py     stuck detection
├── guardrails.py  path confinement, budgets, approval gates
├── trace.py       JSONL run recording and replay
└── cli.py         run / replay / tools
```
