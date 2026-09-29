# Scientific validation evidence

This directory holds dated, human-readable measurements that support scientific or
numerical claims in current DER/RES records and release gates. Each report should
state the command, inputs, comparison and tolerance, observed result, and what the
measurement does not establish. Tests and backend agreement alone do not promote a
derivation or close independent physics acceptance.

Commit a raw JSON packet only when it is needed to inspect or reproduce a claim
that is not recoverable from the concise report, or when a current DER/RES record
cites that exact packet. Existing historical packets remain where their provenance
is cited. Put new large or exploratory packets under ignored `output/validation/`
and summarize durable conclusions here with a reproduction command. Remove a packet
only after its surviving report or record retains its needed evidence.

Start with the [latest production comparison](delta-production-2026-09-29.md) and
[CUDA release gate](cupy-release-2026-09-28.md). Their numerical checks have
explicit scientific coverage limits; consult the corresponding open GitHub issues
for work still needed.
