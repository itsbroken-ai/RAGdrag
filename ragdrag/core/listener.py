"""R3 Support: HTTP Listener for Credential Capture.

Used with RD-0304 (URL Fetcher Exploitation) and RD-0403 (Credential Trap).
Logs all incoming requests and highlights captured credentials.

ATLAS Tactic: Exfiltration

For authorized security testing and research only.
"""

from __future__ import annotations

import json
import fcntl
import ipaddress
import os
import re
import socket
import ssl
import stat
import subprocess
import tempfile
import threading
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, parse_qsl, urlencode, urlparse

from ragdrag.engine.redaction import REDACTED


MAX_BODY_BYTES = 1_048_576
MAX_CONCURRENCY = 64
_CAPTURE_LOCK = threading.Lock()
_PATH_TYPE = type(Path())
_REDACTION_MARKERS = (
    REDACTED, "<withheld>", "hidden", "masked", "0", "1", "◊", "∅",
)


# --- ANSI colors ---

class _Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


# --- Credential detection ---

# Keyword patterns to scan for in query params, headers, and body
CREDENTIAL_KEYWORDS = [
    "api_key",
    "apikey",
    "api-key",
    "token",
    "password",
    "passwd",
    "secret",
    "authorization",
    "bearer",
    "key=",
    "access_key",
    "secret_key",
    "client_secret",
    "client_id",
]

# Regex patterns for structured credential formats
CREDENTIAL_PATTERNS = [
    # AWS access key IDs
    re.compile(r"(AKIA[0-9A-Z]{16})"),
    # JWT tokens
    re.compile(r"(eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})"),
    # Bearer token in header value
    re.compile(r"Bearer\s+(\S+)", re.IGNORECASE),
    # Generic API key patterns (hex, base64-ish)
    re.compile(r"(?:api[_-]?key|token|secret|password)\s*[=:]\s*(\S+)", re.IGNORECASE),
]


@dataclass
class Credential:
    """A detected credential in a request."""

    source: str  # "query", "header", "body"
    key: str
    value: str


def _without_known_secrets(value: str, secrets: tuple[str, ...], marker: str) -> str:
    for secret in secrets:
        value = value.replace(secret, marker)
    return value


def _safe_header_name(
    original: str, secrets: tuple[str, ...], marker: str,
    reserved: set[str], used: set[str], count: int,
) -> str:
    candidate = _without_known_secrets(original, secrets, marker)
    if candidate not in used and (candidate == original or candidate not in reserved):
        return candidate
    for base in ("header-", "h", "k", "◊", "∅"):
        for index in range(1, count + 2):
            for alternate in (f"{base}{index}", base * index):
                if (alternate not in used and alternate not in reserved
                        and not any(secret in alternate for secret in secrets)):
                    return alternate
    raise ValueError("Invalid captured request.")


@dataclass
class CapturedRequest:
    """A single captured HTTP request."""

    timestamp: str
    source_ip: str
    method: str
    path: str
    query_string: str
    headers: dict[str, str]
    body: str
    credentials: list[Credential] = field(default_factory=list)

    def to_dict(self, include_sensitive: bool = False) -> dict:
        _validate_capture(self)
        if type(include_sensitive) is not bool:
            raise ValueError("Invalid captured request.")
        if include_sensitive:
            result: dict = {
                "timestamp": self.timestamp, "source_ip": self.source_ip,
                "method": self.method, "path": self.path,
                "query_string": self.query_string, "headers": dict(self.headers),
                "body": self.body,
            }
            if self.credentials:
                result["credentials"] = [
                    {"source": c.source, "key": c.key, "value": c.value}
                    for c in self.credentials
                ]
            return result

        secrets = tuple(sorted(
            {credential.value for credential in self.credentials if credential.value},
            key=lambda value: (-len(value), value),
        ))
        reserved = {key for key in self.headers
                    if not any(secret in key for secret in secrets)}
        for marker in _REDACTION_MARKERS:
            if any(secret in marker for secret in secrets):
                continue
            headers: dict[str, str] = {}
            for key in self.headers:
                safe_key = _safe_header_name(
                    key, secrets, marker, reserved, set(headers), len(self.headers),
                )
                headers[safe_key] = marker
            result = {
                "timestamp": _without_known_secrets(self.timestamp, secrets, marker),
                "source_ip": _without_known_secrets(self.source_ip, secrets, marker),
                "method": _without_known_secrets(self.method, secrets, marker),
                "path": _without_known_secrets(self.path, secrets, marker),
                "query_string": urlencode([
                    (_without_known_secrets(key, secrets, marker), marker)
                    for key, _ in parse_qsl(self.query_string, keep_blank_values=True)
                ]),
                "headers": headers,
                "body": marker if self.body else "",
            }
            if self.credentials:
                result["credentials"] = [
                    {"source": _without_known_secrets(c.source, secrets, marker),
                     "key": _without_known_secrets(c.key, secrets, marker),
                     "value": marker}
                    for c in self.credentials
                ]
            serialized = _capture_json(result)
            if not any(secret in serialized for secret in secrets):
                return result
        raise ValueError("Invalid captured request.")


def detect_credentials(
    query_params: dict[str, list[str]],
    headers: dict[str, str],
    body: str,
) -> list[Credential]:
    """Scan request components for credential-like values.

    Checks query parameters, headers, and body against keyword patterns
    and structured credential formats (AWS keys, JWTs, Bearer tokens).

    Args:
        query_params: Parsed query string parameters.
        headers: Request headers as key-value pairs.
        body: Raw request body text.

    Returns:
        List of detected Credential objects.
    """
    creds: list[Credential] = []

    # Check query parameters
    for param_name, values in query_params.items():
        name_lower = param_name.lower()
        for keyword in CREDENTIAL_KEYWORDS:
            if keyword in name_lower or name_lower in keyword:
                for val in values:
                    creds.append(Credential(source="query", key=param_name, value=val))
                break

    # Check headers
    for header_name, header_value in headers.items():
        name_lower = header_name.lower()
        if name_lower in ("authorization", "x-api-key", "x-auth-token"):
            creds.append(Credential(source="header", key=header_name, value=header_value))
            continue
        for keyword in CREDENTIAL_KEYWORDS:
            if keyword in name_lower:
                creds.append(Credential(source="header", key=header_name, value=header_value))
                break

    # Check body with keyword scan
    body_lower = body.lower()
    for keyword in CREDENTIAL_KEYWORDS:
        if keyword.rstrip("=") in body_lower:
            # Try to extract key=value pairs from body
            pattern = re.compile(
                rf'["\']?({re.escape(keyword.rstrip("="))}[^"\']*?)["\']?\s*[=:]\s*["\']?([^"\'&\s,}}]+)',
                re.IGNORECASE,
            )
            for match in pattern.finditer(body):
                creds.append(Credential(source="body", key=match.group(1), value=match.group(2)))

    # Check all components with structured patterns
    combined = " ".join(
        list(query_params.keys())
        + [v for vals in query_params.values() for v in vals]
        + list(headers.values())
        + [body]
    )
    for pattern in CREDENTIAL_PATTERNS:
        for match in pattern.finditer(combined):
            value = match.group(1) if match.lastindex else match.group(0)
            # Avoid duplicates
            if not any(c.value == value for c in creds):
                creds.append(Credential(source="pattern", key=pattern.pattern[:30], value=value))

    return creds


def _exact_json(value: object) -> bool:
    if type(value) in (str, int, float, bool, type(None)):
        return True
    if type(value) is list:
        return all(_exact_json(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _exact_json(item) for key, item in value.items())
    return False


def _capture_json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _validate_capture(capture: object) -> None:
    if type(capture) is not CapturedRequest:
        raise ValueError("Invalid captured request.")
    state = object.__getattribute__(capture, "__dict__")
    text_fields = ("timestamp", "source_ip", "method", "path", "query_string", "body")
    if type(state) is not dict or any(
        name not in state or type(state[name]) is not str for name in text_fields
    ):
        raise ValueError("Invalid captured request.")
    if "headers" not in state or type(state["headers"]) is not dict or any(
        type(key) is not str or type(value) is not str
        for key, value in state["headers"].items()
    ):
        raise ValueError("Invalid captured request.")
    if "credentials" not in state or type(state["credentials"]) is not list:
        raise ValueError("Invalid captured request.")
    for credential in state["credentials"]:
        if type(credential) is not Credential:
            raise ValueError("Invalid captured request.")
        fields = object.__getattribute__(credential, "__dict__")
        if type(fields) is not dict or any(
            name not in fields or type(fields[name]) is not str
            for name in ("source", "key", "value")
        ):
            raise ValueError("Invalid captured request.")


def _open_capture_file(output_path: str | Path) -> int:
    path = Path(output_path)
    parts = path.parts
    if not parts or any(part == ".." for part in parts):
        raise OSError("Capture storage failed")
    directory_fd = os.open("/" if path.is_absolute() else ".", os.O_RDONLY | os.O_DIRECTORY)
    fd: int | None = None
    try:
        for part in parts[1 if path.is_absolute() else 0:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=directory_fd)
            old_fd = directory_fd
            directory_fd = next_fd
            os.close(old_fd)
        fd = os.open(parts[-1], os.O_WRONLY | os.O_APPEND | os.O_CREAT |
                     os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
                     0o600, dir_fd=directory_fd)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise OSError("Capture storage failed")
        os.fchmod(fd, 0o600)
        closing_directory_fd = directory_fd
        directory_fd = None
        os.close(closing_directory_fd)
        result = fd
        fd = None
        return result
    finally:
        try:
            if fd is not None:
                closing_fd = fd
                fd = None
                os.close(closing_fd)
        finally:
            if directory_fd is not None:
                closing_directory_fd = directory_fd
                directory_fd = None
                os.close(closing_directory_fd)


def write_capture(
    capture: CapturedRequest, output_path: str | Path, *,
    store_raw: bool = False,
) -> None:
    """Append a captured request to the JSON log file.

    Each capture is written as a single JSON line (JSONL format).

    Args:
        capture: The captured request to log.
        output_path: Path to the output file.
    """
    try:
        if type(store_raw) is not bool:
            raise ValueError("invalid raw selection")
        if type(output_path) is not str and type(output_path) is not _PATH_TYPE:
            raise ValueError("invalid capture path")
        _validate_capture(capture)
        data = CapturedRequest.to_dict(capture, include_sensitive=store_raw)
        if not _exact_json(data):
            raise ValueError("invalid capture")
        line = (_capture_json(data) + "\n").encode("ascii")
        with _CAPTURE_LOCK:
            fd = _open_capture_file(output_path)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                previous_size = os.fstat(fd).st_size
                written = os.write(fd, line)
                if written != len(line):
                    os.ftruncate(fd, previous_size)
                    raise OSError("short capture write")
            finally:
                closing_fd = fd
                fd = None
                os.close(closing_fd)
    except (OSError, ValueError, TypeError, UnicodeError, OverflowError,
            RecursionError, AttributeError):
        raise OSError("Capture storage failed") from None


def _terminal_safe(value: str) -> str:
    safe = []
    for character in value:
        code = ord(character)
        if code < 32 or 127 <= code <= 159:
            safe.append(f"\\x{code:02x}")
        elif unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}:
            safe.append(f"\\u{code:04x}" if code <= 0xFFFF else f"\\U{code:08x}")
        else:
            safe.append(character)
    return "".join(safe)


def format_request_output(capture: CapturedRequest) -> str:
    """Format a captured request for terminal display.

    Normal requests are displayed in dim text. Credential captures
    are highlighted in red with a [!] CAPTURE prefix.

    Args:
        capture: The captured request to format.

    Returns:
        Formatted string for terminal output.
    """
    _validate_capture(capture)
    c = _Colors
    separator = "\u2500" * 45

    lines = [separator]

    has_creds = len(capture.credentials) > 0
    safe = CapturedRequest.to_dict(capture)
    timestamp_value = safe["timestamp"]
    timestamp = timestamp_value.split("T")[1].split(".")[0] if "T" in timestamp_value else timestamp_value
    timestamp = _terminal_safe(timestamp)
    method = _terminal_safe(safe["method"])
    path = _terminal_safe(safe["path"])
    query = _terminal_safe(safe["query_string"])

    if has_creds:
        lines.append(
            f"{c.RED}{c.BOLD}[!] CAPTURE{c.RESET} "
            f"[{timestamp}] {method} {path}"
            f"{'?' + query if query else ''} HTTP/1.1"
        )
    else:
        lines.append(
            f"{c.DIM}[{timestamp}] {method} {path}"
            f"{'?' + query if query else ''} HTTP/1.1{c.RESET}"
        )

    lines.append(f"  From:  {_terminal_safe(safe['source_ip'])}")

    user_agent = safe["headers"].get("User-Agent", safe["headers"].get("user-agent", ""))
    if user_agent:
        lines.append(f"  Agent: {_terminal_safe(user_agent)}")

    if has_creds:
        for cred in safe["credentials"]:
            lines.append(
                f"  {c.RED}{c.BOLD}>> CREDENTIAL: {_terminal_safe(cred['key'])} = "
                f"{_terminal_safe(cred['value'])}{c.RESET}"
            )

    lines.append(separator)
    rendered = "\n".join(lines)
    if any(credential.value and credential.value in rendered
           for credential in capture.credentials):
        raise ValueError("Invalid captured request.")
    return rendered


# --- HTTP Server ---

class _CaptureHandler(BaseHTTPRequestHandler):
    """HTTP request handler that logs all requests and detects credentials."""

    output_path: str = "captures.json"
    max_body_bytes: int = MAX_BODY_BYTES
    store_raw: bool = False

    def _respond(self, status: int, body: bytes = b"") -> None:
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD" and body:
            self.wfile.write(body)

    def _declared_body_length(self) -> int | None:
        lengths = self.headers.get_all("Content-Length", [])
        transfers = self.headers.get_all("Transfer-Encoding", [])
        if transfers or len(lengths) > 1:
            self._respond(400)
            return None
        if not lengths:
            if self.command in {"POST", "PUT", "PATCH", "DELETE"}:
                self._respond(411)
                return None
            return 0
        raw = lengths[0]
        if re.fullmatch(r"[0-9]+", raw) is None:
            self._respond(400)
            return None
        normalized = raw.lstrip("0") or "0"
        maximum = str(self.max_body_bytes)
        if len(normalized) > len(maximum) or (
            len(normalized) == len(maximum) and normalized > maximum
        ):
            self._respond(413)
            return None
        return int(normalized)

    def _handle_request(self) -> None:
        content_length = self._declared_body_length()
        if content_length is None:
            return
        parsed = urlparse(self.path)
        query_params = parse_qs(parsed.query)

        # Read body
        try:
            self.connection.settimeout(5)
            body_bytes = self.rfile.read(content_length) if content_length else b""
        except (OSError, TimeoutError):
            self._respond(400)
            return
        if len(body_bytes) != content_length:
            self._respond(400)
            return
        body = body_bytes.decode("utf-8", errors="replace")

        # Collect headers
        headers = {k: v for k, v in self.headers.items()}

        # Detect credentials
        creds = detect_credentials(query_params, headers, body)

        capture = CapturedRequest(
            timestamp=datetime.now(timezone.utc).isoformat(),
            source_ip=self.client_address[0],
            method=self.command,
            path=parsed.path,
            query_string=parsed.query,
            headers=headers,
            body=body,
            credentials=creds,
        )

        try:
            rendered = format_request_output(capture)
            write_capture(capture, self.output_path, store_raw=self.store_raw)
        except (OSError, ValueError):
            self._respond(500)
            return
        print(rendered)
        self._respond(200, b"OK\n")

    def do_GET(self) -> None:
        self._handle_request()

    def do_POST(self) -> None:
        self._handle_request()

    def do_PUT(self) -> None:
        self._handle_request()

    def do_DELETE(self) -> None:
        self._handle_request()

    def do_PATCH(self) -> None:
        self._handle_request()

    def do_HEAD(self) -> None:
        self._handle_request()

    def do_OPTIONS(self) -> None:
        self._handle_request()

    def log_message(self, format: str, *args: object) -> None:
        # Suppress default http.server logging; we handle our own output
        pass


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """Bound active request workers before a thread is created."""

    request_queue_size = 16
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], handler_class: type,
                 *, max_concurrency: int) -> None:
        if type(max_concurrency) is not int or not 1 <= max_concurrency <= MAX_CONCURRENCY:
            raise ValueError("invalid listener limits")
        self._slots = threading.BoundedSemaphore(max_concurrency)
        super().__init__(server_address, handler_class)

    def process_request(self, request: socket.socket, client_address: tuple) -> None:
        if not self._slots.acquire(blocking=False):
            try:
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\n"
                    b"Connection: close\r\nContent-Length: 0\r\n\r\n"
                )
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            self.shutdown_request(request)
            raise

    def process_request_thread(self, request: socket.socket, client_address: tuple) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
        return address.is_loopback and not (
            isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None
        )
    except ValueError:
        return False


def create_listener_server(
    host: str = "127.0.0.1", port: int = 8443, output: str = "captures.json",
    *, max_body_bytes: int = MAX_BODY_BYTES, max_concurrency: int = 4,
    store_raw: bool = False, allow_public: bool = False,
) -> BoundedThreadingHTTPServer:
    if type(allow_public) is not bool:
        raise ValueError("invalid listener acknowledgement")
    if not _is_loopback_host(host) and not allow_public:
        raise ValueError("non-loopback listener binding requires --allow-public")
    if (type(max_body_bytes) is not int or not 1 <= max_body_bytes <= MAX_BODY_BYTES
            or type(max_concurrency) is not int or not 1 <= max_concurrency <= MAX_CONCURRENCY):
        raise ValueError("invalid listener limits")

    class CaptureHandler(_CaptureHandler):
        pass

    CaptureHandler.output_path = output
    CaptureHandler.store_raw = store_raw
    CaptureHandler.max_body_bytes = max_body_bytes

    server_class = BoundedThreadingHTTPServer
    if ":" in host and not host.startswith("["):
        class IPv6BoundedThreadingHTTPServer(BoundedThreadingHTTPServer):
            address_family = socket.AF_INET6

        server_class = IPv6BoundedThreadingHTTPServer
    return server_class((host, port), CaptureHandler, max_concurrency=max_concurrency)


class OpenSSLUnavailable(RuntimeError):
    """Raised when openssl is missing or fails during self-signed cert generation."""


def generate_self_signed_cert(cert_dir: str | None = None) -> tuple[str, str]:
    """Generate a self-signed TLS certificate and key.

    Uses openssl to create a temporary cert/key pair for HTTPS serving.

    Args:
        cert_dir: Directory to store cert files. Uses tempdir if None.

    Returns:
        Tuple of (cert_path, key_path).

    Raises:
        OpenSSLUnavailable: If openssl is not on PATH or cert generation fails.
    """
    if cert_dir is None:
        cert_dir = tempfile.mkdtemp(prefix="ragdrag-tls-")
    cert_path = str(Path(cert_dir) / "cert.pem")
    key_path = str(Path(cert_dir) / "key.pem")

    try:
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", key_path, "-out", cert_path,
                "-days", "1", "-nodes",
                "-subj", "/CN=ragdrag-listener",
            ],
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as e:
        raise OpenSSLUnavailable(
            "openssl not found on PATH. Install it (macOS: 'brew install openssl'; "
            "Debian/Ubuntu: 'apt install openssl') or run the listener without --tls."
        ) from e
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        raise OpenSSLUnavailable(
            f"openssl failed to generate self-signed cert (exit {e.returncode}). "
            f"stderr: {stderr.strip() or '<empty>'}"
        ) from e

    return cert_path, key_path


def start_listener(
    host: str = "127.0.0.1",
    port: int = 8443,
    output: str = "captures.json",
    tls: bool = False,
    *,
    max_body_bytes: int = MAX_BODY_BYTES,
    max_concurrency: int = 4,
    store_raw: bool = False,
    allow_public: bool = False,
) -> None:
    """Start the credential capture HTTP listener.

    Binds to the specified host/port and logs all incoming requests.
    Credential-bearing requests are highlighted in the terminal output.

    Args:
        host: Host to bind to.
        port: Port to listen on.
        output: Path to the capture log file.
        tls: If True, generate a self-signed cert and serve HTTPS.
    """
    server = create_listener_server(
        host, port, output, max_body_bytes=max_body_bytes,
        max_concurrency=max_concurrency, store_raw=store_raw,
        allow_public=allow_public,
    )

    try:
        scheme = "http"
        if tls:
            cert_path, key_path = generate_self_signed_cert()
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(cert_path, key_path)
            server.socket = ctx.wrap_socket(server.socket, server_side=True)
            scheme = "https"
            print(f"[*] TLS enabled (self-signed cert: {_terminal_safe(cert_path)})")

        print(f"[*] RAGdrag listener active on {scheme}://{_terminal_safe(host)}:{port}")
        print(f"[*] Captures will be saved to {_terminal_safe(output)}")
        print("[*] Press Ctrl+C to stop")
        print()

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[*] Listener stopped.")
    finally:
        server.server_close()
