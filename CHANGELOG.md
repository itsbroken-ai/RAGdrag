# Changelog

## v0.6.0 - Trust Foundation (2026-09-30)

### Breaking safety semantics

- Default `scan` selects R1,R2,R3 at an active-non-mutating ceiling. R6 is opt-in;
  R4/R5 in `scan` require `--allow-write`.
- R4/R5 require a safe cleanup DELETE URL ending in `/{id}` and explicitly
  established baseline, negative-control, and cleanup-verification controls.
  Standalone `poison`/`hijack` retain write intent but require the same controls.
- Existing command names and options remain compatible. Scripts relying on all
  six default phases or untracked writes must adopt explicit safety inputs.

### Added

- Typed capabilities, origin-bound transport, request/response budgets,
  evidence states, and a mutation ledger with cleanup on failure/interruption.
- Schema 1.0 reports and exit codes 0–5 for findings, partial assessments, and
  unresolved cleanup. Default reports redact credentials and raw extraction.
- Explicit chat history/session flags and scoped usable-cookie state.
- Loopback listener default, public-bind acknowledgment, body/worker bounds,
  escaped terminal output, and owner-only raw-capture opt-in.
- Generated capability status from executable metadata, with drift checks.
- Python 3.10/3.11/3.12 CI gates, wheel/sdist builds, and clean-install checks;
  packages include report schema, payload JSON, and `py.typed`.

### Limits

- DELETE 202 establishes acceptance, not eventual absence, even though the
  current handler records it as `removed`. Generic POST upsert safety is not
  proven without create-only or restore semantics.
- Listener bounds cover body bytes and worker count; they do not establish
  absolute connection lifetime, slow header/body progress deadlines, TLS
  handshake deadlines, or daemon-worker completion on shutdown.
- Registry maturity and local tests do not certify arbitrary target findings.
  Catalogued taxonomy entries are not implementation claims.

## v0.5.0 - Full Kill Chain (2026-04-01)

### Added
- **R4 Poison phase** with 4 techniques:
  - RD-0401: Document Injection (ingestion endpoint discovery, injection, verification)
  - RD-0402: Embedding Dominance (test if injected docs dominate retrieval)
  - RD-0403: Credential Trap (inject docs directing users to attacker infrastructure)
  - RD-0404: Instruction Injection via Retrieval (inject directives that influence LLM behavior)
- **R5 Hijack phase** with 4 techniques:
  - RD-0501: Retrieval Redirection (replace expected responses with attacker content)
  - RD-0502: Context Window Saturation (flood context with attacker documents)
  - RD-0503: Agent Tool Manipulation (inject docs that trigger tool calls)
  - RD-0504: Persistent Backdoor via RAG (verify persistence across query types)
- **R6 Evade phase** with 4 techniques:
  - RD-0601: Semantic Substitution (3 strategies: academic, business, indirect)
  - RD-0602: Retrieval Camouflage (wrap payloads in organizational content)
  - RD-0603: Query Pattern Obfuscation (noise interleaving)
  - RD-0604: Multi-Turn Context Building (progressive disclosure, role assumption)
- CLI commands: `ragdrag poison`, `ragdrag hijack`, `ragdrag evade`
- `ragdrag scan` now chains all 6 phases (R1-R6) with JSON output
- Cross-phase composition: Hijack imports from Poison + Evade
- pytest testpaths configuration

### Changed
- Renamed `test_*` functions in source modules to `assess_*` to prevent pytest collection conflicts
- `ragdrag scan` default phases expanded from R1,R2,R3 to R1,R2,R3,R4,R5,R6

## v0.2.0 - R2 Probe (2026-04-01)

### Added
- **R2 Probe phase** with 5 techniques for mapping RAG pipeline internals:
  - RD-0201: Chunk Boundary Detection (source counts, chunk sizes, cross-topic analysis, response variation)
  - RD-0202: Similarity Threshold Mapping (7-level relevance degradation, cutoff detection)
  - RD-0203: Retrieval Count Estimation (top-k detection, fixed vs dynamic)
  - RD-0204: Knowledge Base Scope Mapping (6 domain categories, coverage/gap analysis)
  - RD-0205: Embedding Model Fingerprinting (multilingual, code-aware, domain-specific, general classification)
- Debug endpoint discovery (13 common paths: /debug/config, /admin/stats, etc.)
- `ragdrag probe` CLI command with `--depth quick|full`
- `ragdrag scan` command chaining R1, R2, R3 phases
- Shared data structures in `ragdrag/core/models.py` (Finding dataclass)
- Payload loader utility (`ragdrag/utils/payloads.py`)
- Generalized JSON reporter (accepts any result with `.to_dict()`)
- ProbeResult dataclass with `.to_dict()` serialization
- 70 tests (probe techniques, shared models, payload loader)

### Changed
- Extracted Finding from fingerprint.py to shared models.py
- JSON reporter now accepts any phase result, not just FingerprintResult

## v0.1.0-alpha - Initial Release (2026-03-25)

### Added
- R1 Fingerprint phase (RD-0101: RAG presence detection, RD-0102: Vector DB fingerprinting)
- R3 Exfiltrate phase (RD-0301: Direct knowledge extraction, RD-0302: Guardrail-aware extraction)
- RAGdrag taxonomy: 27 techniques across 6 phases
- CLI: `ragdrag fingerprint`, `ragdrag exfiltrate`, `ragdrag listen`
- JSON reporter for findings output
- Lab servers (open + guarded RAG targets)
- 7 payload files for curated query sets
