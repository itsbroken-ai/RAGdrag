<h1 align="center">RAGdrag</h1>

<p align="center">
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
<img src="https://img.shields.io/badge/version-0.6.0-orange.svg" alt="Version 0.6.0">
</p>

RAG pipeline security assessment toolkit for authorized testing.

RAGdrag assesses retrieval, generation, and ingestion boundaries through six
phases. The executable engine registry currently exposes 20 technique IDs. The
27-entry taxonomy also contains catalogued ideas without executable phase
claims. See the generated [implementation status](docs/implementation-status.md)
for capability IDs, technique IDs, impact, and maturity.

## Installation

```bash
git clone https://github.com/McKern3l/RAGdrag.git
cd RAGdrag
python -m pip install -e .
ragdrag --version
```

Requires Python 3.10 or later. CI gates Python 3.10, 3.11, and 3.12.

## Quick start

Use an endpoint you own or are explicitly authorized to assess. These examples
use a local lab. Default `scan` selects R1,R2,R3 at an `active-non-mutating`
impact ceiling. It sends requests and can expose sensitive responses; R4/R5
ingestion writes require explicit authorization. R6 is opt-in.

```bash
ragdrag scan --target http://127.0.0.1:8899/chat --output report.json
```

Explicitly include R6 when the scope includes evasion assessment:

```bash
ragdrag scan --target http://127.0.0.1:8899/chat --phases R1,R2,R3,R6 \
  --history-field messages --response-field response --output report.json
```

Existing command names and options remain compatible. The default scan and
write requirements changed in 0.6.0; scripts may need to select phases and
establish cleanup controls explicitly.

## Write authorization and cleanup

R4/R5 require a cleanup DELETE URL with exactly one final `/{id}` path segment,
no query, fragment, or userinfo, and an approved origin. For `scan`, also use
`--allow-write`. Standalone `poison` and `hijack` express write intent themselves
and still require cleanup and all three controls.

Establish a baseline, a negative control, and cleanup verification in your lab
before asserting them with `--established-control`. These flags declare existing
operational controls; they do not perform or prove those controls.

```bash
ragdrag scan --target http://127.0.0.1:8899/chat --phases R4,R5 --allow-write \
  --cleanup-url 'http://127.0.0.1:8899/documents/{id}' \
  --established-control baseline \
  --established-control negative-control \
  --established-control cleanup-verification --output write-report.json
```

The mutation ledger tracks write attempts and runs registered cleanup on normal,
failed, and interrupted execution. Unknown creation outcomes and failed cleanup
remain unresolved. Inspect mutation records and verify target state after the
run. The current DELETE handler records HTTP 200, 202, 204, or 404 as `removed`;
**202 establishes acceptance, not eventual absence**. Generic POST ingestion may
upsert an existing object: safety is not proven without a create-only contract
or restore semantics. A deletion route alone does not establish that contract.

## Credentials and conversation state

`--header` / `-H` and `--cookie` are scoped to the exact target origin: scheme,
hostname, and effective port. Alternate ports and cross-origin redirects do not
inherit credentials. An independently approved ingestion origin receives only
its own configured headers. `poison` and `hijack` accept `--ingest-url` and
`--api-key`; that key is scoped to the ingestion origin and does not migrate to
the chat origin on redirects. TLS verification is enabled by default;
`--no-verify-ssl` explicitly disables it.

Network commands share `--query-field` (default `query`), `--response-field`,
`--history-field`, `--session-field`, and `--session-id`. A configured response
field must contain a JSON string; malformed or missing data produces an
`unsupported-response` outcome. Without a response field, response text is used.

Use `--history-field messages` for a target that accepts conversation messages,
or `--session-field session_id --session-id lab-session` for an explicit session.
The adapter also tracks usable cookies established by the target. Multi-turn
assessment requires history, an explicit session, or an established usable
cookie; without real state it reports `capability-not-applicable`.

## Reports and exit codes

Reports use schema version **1.0**, shipped at
[`ragdrag/reporters/schemas/report-v1.schema.json`](ragdrag/reporters/schemas/report-v1.schema.json).
They include target configuration, run status, capabilities, findings, evidence,
mutations, implementation status, and summary counts. Finding evidence states
are `observed`, `inferred`, and `validated`; these describe evidence, separately
from capability maturity. A heuristic finding alone does not prove a
vulnerability. Assess authentication failures, unsupported responses, and
indeterminate execution alongside findings.

Default reports withhold raw extraction text and redact credentials and untrusted
free text. Validate a saved schema 1.0 report with the established command:

```bash
ragdrag report --input report.json --format json
```

| Code | Meaning |
|---|---|
| 0 | Completed cleanly, no findings |
| 1 | Completed with findings |
| 2 | Partial, blocked, interrupted, or indeterminate assessment |
| 3 | Invalid configuration or target |
| 4 | Execution failure |
| 5 | Unresolved cleanup; takes precedence over findings and other failures |

A blocked or unreachable target is not a clean assessment. Preserve reports on
nonzero exits; they can contain partial evidence and cleanup state.

## Listener

The listener binds to `127.0.0.1` by default. Capture output is JSON Lines;
credentials and raw request content are withheld by default and untrusted
terminal text is escaped. Capture files use owner-only mode `0600` and reject
unsafe aliases. Raw request storage is a sensitive opt-in:

```bash
ragdrag listen --port 8443 --output captures.jsonl
ragdrag listen --port 8443 --store-raw --output private-captures.jsonl
```

Protect raw captures and follow your engagement retention policy. Non-loopback
binding requires `--allow-public`; `--tls` enables the existing self-signed TLS
mode. `--max-body-bytes` defaults to 1,048,576 and accepts 1–1,048,576;
`--max-concurrency` defaults to 4 and accepts 1–64. Unsupported framing and
oversized declared bodies are rejected, and saturated workers return 503. These
bounds do not establish absolute connection lifetime, slow header/body progress
deadlines, TLS handshake deadlines, or daemon-worker completion on shutdown.
Use infrastructure appropriate to your scope when those guarantees are needed.

## Commands and capability claims

| Command | Behavior |
|---|---|
| `fingerprint` | R1: RAG presence and vector database assessment; `--no-port-scan` available |
| `probe` | R2: Retrieval and knowledge scope assessment; `--depth quick` or `full` |
| `exfiltrate` | R3: Knowledge exposure assessment; `--deep` available |
| `poison` | R4: Ingestion assessment with explicit cleanup controls |
| `hijack` | R5: Retrieval/generation assessment with explicit cleanup controls |
| `evade` | R6: Evasion assessment, including real conversation state |
| `scan` | Selected phases; defaults to R1,R2,R3 |
| `listen` | Local capture listener with protected raw-storage opt-in |
| `report` | Validate and display a schema 1.0 JSON report |

Use `ragdrag COMMAND --help` for the complete option contract. R2 executable
names are chunk boundary detection (RD-0201), similarity threshold mapping
(RD-0202), retrieval count estimation (RD-0203), knowledge base scope mapping
(RD-0204), and embedding model fingerprinting (RD-0205). Historical taxonomy
names differ; executable metadata and source define current behavior.

Catalogued entries without independent executable registry claims are RD-0103,
RD-0104, RD-0105, RD-0303, RD-0304, RD-0305, and RD-0602. Camouflage helpers may be
used by other phases, but RD-0602 is not an R6 registry technique in this release.

Maturity labels mean: `catalogued` describes an idea; `implemented` describes
executable registry behavior; `validated` requires a declared test file present
in the checkout; `experimental` retains the registry's qualification. The
generated table does not certify target-specific findings or prove that tests
ran. CI runs the suite and checks the artifact for drift.

## Development and release checks

```bash
python -m pip install -e '.[dev]'
pytest -q
python -m ragdrag.engine.status --check docs/implementation-status.md
python -m build
```

After changing executable metadata, regenerate with
`python -m ragdrag.engine.status --write docs/implementation-status.md` from the
checkout root. Validation paths are checkout-relative: installed packages
without the test suite fall back to `implemented`. CI checks and regenerates the
artifact, builds wheel and source distribution, and smoke-tests a fresh wheel
installation on Python 3.10, 3.11, and 3.12.

## Lab, contributions, and license

The optional [RAGdrag labs](https://github.com/McKern3l/RAGdrag-labs) provide local
targets and exercises. Contributions should describe reproducible behavior,
validation evidence, and limits. Use RAGdrag only within explicit authorization
and disclose findings through appropriate channels.

MIT — see [LICENSE](LICENSE). Author: [McKern3l](https://github.com/McKern3l).
