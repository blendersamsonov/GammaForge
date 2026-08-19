# GammaForge — Postmortems

A postmortem is **not** a [decision](../decisions/README.md) (which records a deliberate
design or implementation choice and its rejected alternatives, or proposes future work).
It is a backward-looking record of a failure: a bug reached a place it shouldn't have (a
real user, a merged PR, a release), and the interesting part is *why the process let it
through*, not just the one-line fix.

Write one when a bug is:

- **subtle** — the mechanism is non-obvious, and a careful person would re-derive it the
  hard way;
- **systemic** — the reason it escaped is a gap in tests, tooling, or convention, not a
  one-off typo; and
- **costly to rediscover** — it burned real debugging time, and would burn it again.

Link the guardrails the postmortem motivated (new tests, an `AGENTS.md` rule, a decision
record) at the end.

Every postmortem opens with an **Executive summary**: one short paragraph a busy reader
can absorb in thirty seconds — what broke, the root cause in plain terms, why it escaped,
and the durable lesson — before the detailed Summary / Timeline / Root cause / Guardrails
sections that follow.

| # | Title |
|---|-------|
