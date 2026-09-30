"""Command-line adapter for the origin-bound engagement engine."""

from __future__ import annotations

import json
import logging
import os
import re
import stat
import sys
from pathlib import Path
from urllib.parse import urlparse

import click

from ragdrag import __version__
from ragdrag.core.poison import CleanupStrategy
from ragdrag.core.listener import MAX_BODY_BYTES, MAX_CONCURRENCY, _is_loopback_host
from ragdrag.engine.capability import InvalidConfiguration
from ragdrag.engine.models import ExitCode, ImpactLevel, RequestBudget
from ragdrag.engine.phases import (
    MAX_CLEANUP_URL_LENGTH, EngagementOutcome, PhaseOptions, run_engagement,
)
from ragdrag.engine.profile import TargetProfile, canonical_origin, normalize_target
from ragdrag.engine.runner import EngagementInterrupted
from ragdrag.reporters.json_report import _validate_report, format_summary, generate_run_report


BANNER = click.style(
    "    ┌──────────────────────────────────┐\n"
    f"    │  RAGdrag v{__version__:<23s}│\n"
    "    │  RAG Pipeline Security Toolkit   │\n"
    "    │  github.com/McKern3l             │\n"
    "    └──────────────────────────────────┘\n",
    fg="cyan",
)
DEFAULT_SCAN_PHASES = "R1,R2,R3"
MUTATING_PHASES = frozenset({"R4", "R5"})
KNOWN_PHASES = frozenset({"R1", "R2", "R3", "R4", "R5", "R6"})
REQUIRED_CONTROLS = frozenset({"baseline", "negative-control", "cleanup-verification"})
_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+\Z")
MAX_FIELD_NAME_BYTES = 256


class ConfigurationError(click.UsageError):
    """A fixed, safe message for rejected local configuration."""


class RagdragGroup(click.Group):
    def main(self, *args: object, **kwargs: object) -> object:
        kwargs["standalone_mode"] = False
        try:
            result = super().main(*args, **kwargs)
            if type(result) is int:
                raise SystemExit(result)
            return result
        except click.exceptions.Exit as error:
            raise SystemExit(error.exit_code) from None
        except ConfigurationError as error:
            click.echo(f"Error: {error.message}", err=True)
            raise SystemExit(ExitCode.INVALID_CONFIGURATION) from None
        except click.ClickException:
            click.echo("Error: Invalid command options.", err=True)
            raise SystemExit(ExitCode.INVALID_CONFIGURATION) from None


def _has_control(value: str) -> bool:
    return any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)


def _valid_http_header_text(value: object) -> bool:
    """Match HTTPX's ASCII string-header contract, allowing horizontal tabs."""
    if type(value) is not str:
        return False
    try:
        value.encode("ascii")
    except UnicodeError:
        return False
    return not any((ord(char) < 32 and char != "\t") or ord(char) == 127 for char in value)


def _valid_json_field(value: object) -> bool:
    if type(value) is not str or not value or _has_control(value):
        return False
    try:
        return len(value.encode("utf-8")) <= MAX_FIELD_NAME_BYTES
    except UnicodeError:
        return False


def _validate_output_destination(output: str | None) -> None:
    if output is None:
        return
    if type(output) is not str or not output or "\x00" in output:
        raise ConfigurationError("Invalid --output destination.")
    try:
        path = Path(output)
        nearest_parent: Path | None = None
        for parent in path.parents:
            try:
                mode = parent.stat().st_mode
            except FileNotFoundError:
                try:
                    parent.lstat()
                except FileNotFoundError:
                    continue
                raise ConfigurationError("Invalid --output destination.") from None
            if nearest_parent is None:
                nearest_parent = parent
            if not stat.S_ISDIR(mode):
                raise ConfigurationError("Invalid --output destination.")
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            if nearest_parent is None or not os.access(
                nearest_parent, os.W_OK | os.X_OK, effective_ids=True,
            ):
                raise ConfigurationError("Invalid --output destination.")
            return
        if not stat.S_ISREG(mode) or not os.access(path, os.W_OK, effective_ids=True):
            raise ConfigurationError("Invalid --output destination.")
    except (OSError, ValueError, UnicodeError, TypeError, NotImplementedError):
        raise ConfigurationError("Invalid --output destination.") from None


def _validate_url(url: str) -> str:
    """Normalize a safe HTTP URL without reflecting rejected input."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise click.BadParameter("Invalid URL: must start with http:// or https://")
        if not parsed.netloc or not parsed.hostname:
            raise click.BadParameter("Invalid URL: missing host")
        if (
            _has_control(url) or any(char.isspace() for char in url)
            or "\\" in url or parsed.username is not None or parsed.password is not None
            or "@" in parsed.netloc or parsed.port == 0
        ):
            raise click.BadParameter("Invalid URL")
        return normalize_target(url)
    except click.BadParameter:
        raise
    except (ValueError, TypeError, OverflowError):
        raise click.BadParameter("Invalid URL") from None


def _parse_headers(header: tuple[str, ...], cookie: str | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    seen: set[str] = set()
    for item in header:
        if not _valid_http_header_text(item) or ":" not in item:
            raise ConfigurationError("Invalid --header value.")
        name, value = item.split(":", 1)
        name, value = name.strip(" \t"), value.strip(" \t")
        lowered = name.lower()
        if (
            _HEADER_NAME.fullmatch(name) is None
            or lowered in seen or lowered in {"host", "content-length", "transfer-encoding"}
        ):
            raise ConfigurationError("Invalid or ambiguous --header value.")
        seen.add(lowered)
        headers[name] = value
    if cookie is not None:
        if not cookie or not _valid_http_header_text(cookie) or "cookie" in seen:
            raise ConfigurationError("Invalid or ambiguous --cookie value.")
    return headers


def _normalized_wire_credential(value: str, option: str) -> str:
    if not _valid_http_header_text(value):
        raise ConfigurationError(f"Invalid {option} value.")
    normalized = value.strip(" \t")
    if not normalized:
        raise ConfigurationError(f"Invalid {option} value.")
    return normalized


def _parse_phases(value: str) -> list[str]:
    parts = value.split(",")
    phases = list(dict.fromkeys(part.strip().upper() for part in parts))
    if any(not part.strip() for part in parts) or any(
        phase not in KNOWN_PHASES for phase in phases
    ):
        raise ConfigurationError("--phases must select one or more of R1,R2,R3,R4,R5,R6.")
    return phases


def _validate_write_request(
    phases: list[str], allow_write: bool, cleanup_url: str | None,
    established_controls: frozenset[str], approved_origins: frozenset[str],
) -> None:
    mutating = MUTATING_PHASES.intersection(phases)
    if mutating and not allow_write:
        raise ConfigurationError("R4/R5 require --allow-write.")
    if mutating and not cleanup_url:
        raise ConfigurationError("R4/R5 require --cleanup-url with a final {id} segment.")
    if mutating and not REQUIRED_CONTROLS.issubset(established_controls):
        raise ConfigurationError(
            "R4/R5 require --established-control baseline, negative-control, "
            "and cleanup-verification."
        )
    if cleanup_url is not None:
        try:
            if type(cleanup_url) is not str or len(cleanup_url) > MAX_CLEANUP_URL_LENGTH:
                raise ValueError("length")
            cleanup = CleanupStrategy(cleanup_url)
            resolved = cleanup.url_for("preflight-id")
            if canonical_origin(resolved) not in approved_origins:
                raise ValueError("origin")
        except (ValueError, TypeError, OverflowError):
            raise ConfigurationError("Invalid or unapproved --cleanup-url.") from None


def _execute_phases(
    *, target: str, phases: list[str], output: str | None, header: tuple[str, ...],
    cookie: str | None, timeout: float, no_verify_ssl: bool, query_field: str,
    response_field: str | None, history_field: str | None, session_field: str | None,
    session_id: str | None, cleanup_url: str | None,
    established_control: tuple[str, ...], allow_write: bool,
    phase_options: dict[str, object] | None = None, api_key: str | None = None,
) -> int:
    target = _validate_url(target)
    headers = _parse_headers(header, cookie)
    if cookie is not None:
        cookie = _normalized_wire_credential(cookie, "--cookie")
    for name, value in (("--query-field", query_field), ("--response-field", response_field),
                        ("--history-field", history_field), ("--session-field", session_field)):
        if value is not None and not _valid_json_field(value):
            raise ConfigurationError(f"Invalid {name} value.")
    if session_id is not None and (not session_id or _has_control(session_id)):
        raise ConfigurationError("Invalid --session-id value.")
    if api_key is not None:
        api_key = _normalized_wire_credential(api_key, "--api-key")
    if any(name.lower() == "x-api-key" for name in headers) and api_key is not None:
        raise ConfigurationError("Ambiguous --api-key and --header configuration.")

    options_data = dict(phase_options or {})
    ingest_url = options_data.get("ingest_url")
    if ingest_url is not None:
        ingest_url = _validate_url(ingest_url)
        options_data["ingest_url"] = ingest_url
    callback_url = options_data.get("callback_url")
    if callback_url is not None:
        options_data["callback_url"] = _validate_url(callback_url)
    listener_host = options_data.get("listener_host")
    if listener_host is not None and (not listener_host or _has_control(listener_host)):
        raise ConfigurationError("Invalid --listener value.")

    primary_origin = canonical_origin(target)
    ingestion_origin = canonical_origin(ingest_url or target)
    additional: dict[str, dict[str, str]] = {}
    if ingestion_origin != primary_origin:
        additional[ingestion_origin] = {}
    if api_key is not None:
        if ingestion_origin == primary_origin:
            headers["X-Api-Key"] = api_key
        else:
            additional[ingestion_origin]["X-Api-Key"] = api_key
    approved_origins = frozenset({primary_origin, ingestion_origin})
    established_controls = frozenset(established_control)
    _validate_write_request(phases, allow_write, cleanup_url, established_controls, approved_origins)
    try:
        profile = TargetProfile.from_cli(
            target, headers=headers, cookie=cookie,
            additional_origin_headers=additional,
            query_field=query_field, response_field=response_field,
            history_field=history_field, session_field=session_field,
            session_id=session_id, verify_ssl=not no_verify_ssl,
            impact_ceiling=ImpactLevel.MUTATING if allow_write else ImpactLevel.ACTIVE,
            budget=RequestBudget(timeout_seconds=timeout),
        )
        options = PhaseOptions(
            query_field=query_field, response_field=response_field,
            history_field=history_field, session_field=session_field,
            session_id=session_id, cleanup_url=cleanup_url,
            established_controls=established_controls, **options_data,
        )
    except (ValueError, TypeError, OverflowError):
        raise ConfigurationError("Invalid target or engagement configuration.") from None
    _validate_output_destination(output)
    previous_logging_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        try:
            outcome = run_engagement(profile, phases, options)
        except EngagementInterrupted as interrupted:
            outcome = EngagementOutcome(interrupted.run, interrupted.evidence, interrupted.mutations)
        except InvalidConfiguration:
            click.echo("Error: Invalid engagement configuration.", err=True)
            return int(ExitCode.INVALID_CONFIGURATION)
        except Exception:
            click.echo("Error: Engagement execution failed.", err=True)
            return int(ExitCode.EXECUTION_FAILURE)
    finally:
        logging.disable(previous_logging_disable)
    try:
        report = generate_run_report(outcome, profile)
        summary = format_summary(report)
    except (ValueError, TypeError, OSError):
        click.echo("Error: Report generation failed.", err=True)
        return int(ExitCode.UNRESOLVED_CLEANUP if outcome.run.exit_code == ExitCode.UNRESOLVED_CLEANUP
                   else ExitCode.EXECUTION_FAILURE)
    click.echo(summary)
    if output is not None:
        try:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_validate_report(report), encoding="utf-8")
        except (OSError, ValueError, TypeError, UnicodeError):
            click.echo("Error: Report output failed.", err=True)
            return int(ExitCode.UNRESOLVED_CLEANUP if outcome.run.exit_code == ExitCode.UNRESOLVED_CLEANUP
                       else ExitCode.EXECUTION_FAILURE)
    return int(outcome.run.exit_code)


def _network_options(function):
    """Apply the same scoped transport and chat-state inputs to every network command."""
    decorators = (
        click.option("--target", "-t", required=True, help="Target RAG endpoint URL."),
        click.option("--query-field", default="query", help="JSON field name for queries."),
        click.option("--response-field", default=None, help="JSON field to read responses from."),
        click.option("--history-field", default=None, help="JSON field containing chat history."),
        click.option("--session-field", default=None, help="JSON field containing session ID."),
        click.option("--session-id", default=None, help="Explicit chat session ID."),
        click.option("--output", "-o", default=None, help="Output file path for JSON report."),
        click.option("--timeout", type=float, default=30.0, help="HTTP request timeout in seconds."),
        click.option("--header", "-H", "header", multiple=True,
                     help="Extra origin-scoped request header 'Key: Value' (repeatable)."),
        click.option("--cookie", default=None, help="Origin-scoped Cookie header value."),
        click.option("--no-verify-ssl", is_flag=True, help="Disable SSL verification."),
        click.option("--cleanup-url", default=None, help="Safe DELETE URL ending in /{id} for R4/R5."),
        click.option("--established-control", multiple=True,
                     type=click.Choice(sorted(REQUIRED_CONTROLS)),
                     help="Assert an already-established operational control (repeatable)."),
    )
    for decorator in reversed(decorators):
        function = decorator(function)
    return function


def _finish(phases: list[str], common: dict[str, object], *, allow_write: bool = False,
            phase_options: dict[str, object] | None = None, api_key: str | None = None) -> None:
    raise click.exceptions.Exit(_execute_phases(
        phases=phases, allow_write=allow_write, phase_options=phase_options,
        api_key=api_key, **common,
    ))


@click.group(cls=RagdragGroup)
@click.version_option(version=__version__, prog_name="ragdrag")
@click.option("--quiet", "-q", is_flag=True, help="Suppress banner output.")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug-level logging.")
def cli(quiet: bool, verbose: bool) -> None:
    """ragdrag - RAG pipeline security testing toolkit.

    Phases: R1 Fingerprint, R2 Probe, R3 Exfiltrate, R4 Poison,
    R5 Hijack, R6 Evade. For authorized security testing and research only.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="[%(levelname)s] %(name)s: %(message)s", stream=sys.stderr,
    )
    if not quiet:
        click.echo(BANNER)


@cli.command()
@click.option("--no-port-scan", is_flag=True, help="Skip vector DB port scanning.")
@_network_options
def fingerprint(no_port_scan: bool, **common: object) -> None:
    """R1: Fingerprint a target for RAG presence and vector DB identification.

    Runs RD-0101 (RAG Presence Detection) and RD-0102 (Vector DB
    Fingerprinting) against the target endpoint.
    """
    _finish(["R1"], common, phase_options={"scan_ports": not no_port_scan})


@cli.command()
@click.option("--depth", type=click.Choice(["quick", "full"]), default="quick",
              help="Probe depth: quick (RD-0201) or full (all R2 techniques).")
@_network_options
def probe(depth: str, **common: object) -> None:
    """R2: Probe RAG pipeline internals.

    Maps chunk boundaries, retrieval parameters, and knowledge base scope.
    Techniques: RD-0201 (Chunk Boundary Detection). More in --depth full.
    """
    _finish(["R2"], common, phase_options={"deep": depth == "full"})


@cli.command()
@click.option("--deep", is_flag=True, help="Enable guardrail bypass techniques (RD-0302).")
@_network_options
def exfiltrate(deep: bool, **common: object) -> None:
    """R3: Extract knowledge base contents, credentials, and sensitive data.

    Runs RD-0301 (Direct Knowledge Extraction) and optionally RD-0302
    (Guardrail-Aware Extraction) with the --deep flag.
    """
    _finish(["R3"], common, phase_options={"deep": deep})


@cli.command()
@click.option("--listener", "-l", default=None, help="Listener host for credential traps.")
@click.option("--ingest-url", default=None, help="Override ingestion endpoint URL.")
@click.option("--api-key", default=None, help="API key scoped to the ingestion origin.")
@_network_options
def poison(listener: str | None, ingest_url: str | None, api_key: str | None,
           **common: object) -> None:
    """R4: Inject cleanable content into the knowledge base."""
    _finish(["R4"], common, allow_write=True, api_key=api_key,
            phase_options={"ingest_url": ingest_url, "listener_host": listener})


@cli.command()
@click.option("--callback", "-c", default=None, help="Callback URL for tool manipulation.")
@click.option("--ingest-url", default=None, help="Override ingestion endpoint URL.")
@click.option("--api-key", default=None, help="API key scoped to the ingestion origin.")
@click.option("--camouflage", is_flag=True, help="Wrap injected docs in R6 camouflage.")
@_network_options
def hijack(callback: str | None, ingest_url: str | None, api_key: str | None,
           camouflage: bool, **common: object) -> None:
    """R5: Take control of RAG pipeline retrieval and generation."""
    _finish(["R5"], common, allow_write=True, api_key=api_key,
            phase_options={"ingest_url": ingest_url, "callback_url": callback,
                           "camouflage": camouflage})


@cli.command()
@_network_options
def evade(**common: object) -> None:
    """R6: Test evasion techniques against guardrails and monitoring."""
    _finish(["R6"], common)


@cli.command()
@click.option("--phases", "-p", default=DEFAULT_SCAN_PHASES,
              help="Comma-separated phases (R1-R6). Default: R1,R2,R3; R6 opt-in.")
@click.option("--allow-write", is_flag=True, help="Explicitly enable R4/R5 mutations.")
@_network_options
def scan(phases: str, allow_write: bool, **common: object) -> None:
    """Run selected RAGdrag phases against a target.

    Phases: R1 (Fingerprint), R2 (Probe), R3 (Exfiltrate),
    R4 (Poison), R5 (Hijack), R6 (Evade).
    """
    _finish(_parse_phases(phases), common, allow_write=allow_write)


def _check_report_controls(value: object) -> None:
    if type(value) is str and _has_control(value):
        raise ValueError("invalid report")
    if type(value) is dict:
        for key, item in value.items():
            _check_report_controls(key)
            _check_report_controls(item)
    elif type(value) is list:
        for item in value:
            _check_report_controls(item)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate report key")
        value[key] = item
    return value


@cli.command()
@click.option("--input", "-i", "input_file", required=True, help="Input findings JSON file.")
@click.option("--format", "-f", "fmt", type=click.Choice(["json"]), default="json",
              help="Output format.")
@click.option("--output", "-o", default=None, help="Output file path.")
def report(input_file: str, fmt: str, output: str | None) -> None:
    """Validate and display a schema 1.0 JSON report."""
    try:
        source = Path(input_file)
        if source.stat().st_size > 10_000_000:
            raise ValueError("large report")
        data = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        _check_report_controls(data)
        serialized = _validate_report(data)
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError):
        raise ConfigurationError("Invalid report file.") from None
    if output:
        try:
            Path(output).write_text(serialized, encoding="utf-8")
        except OSError:
            raise ConfigurationError("Report output failed.") from None
        click.echo("Report written.")
    else:
        click.echo(serialized, nl=False)


@cli.command()
@click.option("--port", "-p", default=8443, help="Port to listen on.")
@click.option("--host", default="127.0.0.1", help="Host to bind to.")
@click.option("--output", "-o", default="captures.json", help="Capture log file.")
@click.option("--tls", is_flag=True, help="Enable TLS with self-signed cert.")
@click.option("--allow-public", is_flag=True, help="Acknowledge non-loopback binding.")
@click.option("--max-body-bytes", type=click.IntRange(1, MAX_BODY_BYTES),
              default=MAX_BODY_BYTES, help="Maximum declared request body size.")
@click.option("--max-concurrency", type=click.IntRange(1, MAX_CONCURRENCY),
              default=4, help="Maximum simultaneous request workers.")
@click.option("--store-raw", is_flag=True, help="Store raw sensitive request data.")
def listen(port: int, host: str, output: str, tls: bool, allow_public: bool,
           max_body_bytes: int, max_concurrency: int, store_raw: bool) -> None:
    """Start a credential capture HTTP listener.

    Logs all incoming HTTP requests and highlights credential captures.
    Used with RD-0304 (URL Fetcher Exploitation) and RD-0403 (Credential Trap).
    """
    from ragdrag.core.listener import start_listener

    if not _is_loopback_host(host) and not allow_public:
        raise ConfigurationError("non-loopback listener binding requires --allow-public")
    start_listener(
        host=host, port=port, output=output, tls=tls,
        max_body_bytes=max_body_bytes, max_concurrency=max_concurrency,
        store_raw=store_raw, allow_public=allow_public,
    )
