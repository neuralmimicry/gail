#!/usr/bin/env python3
"""Download a pinned Hugging Face model for a Gail Slurm training job.

The Slurm job authenticates to Gail with a protected, model-download-only
service token. Gail returns the current provider token just in time; this
process passes it directly to ``huggingface_hub`` and never writes it to the
Slurm environment, scheduler metadata, cache, provenance file or output.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import logging
import os
import re
import stat
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import warnings
from pathlib import Path
from typing import Sequence


MODEL_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
MAX_API_TOKEN_BYTES = 8192
MAX_PROVIDER_TOKEN_BYTES = 8192


class SafeDownloaderError(RuntimeError):
    """An operator-safe error detail that contains no provider response text."""


def safe_failure_detail(error: Exception) -> str:
    """Keep useful status while suppressing third-party exception messages."""
    if isinstance(error, SafeDownloaderError):
        return str(error)
    return {
        "GatedRepoError": "Hugging Face denied access to the configured model",
        "RepositoryNotFoundError": "Hugging Face could not find or access the configured model",
        "RevisionNotFoundError": "Hugging Face could not find the pinned model revision",
        "TimeoutError": "the model provider request timed out",
        "ConnectionError": "the model provider connection failed",
    }.get(type(error).__name__, type(error).__name__)


def validate_model_reference(model_id: str, revision: str) -> None:
    """Require a simple Hugging Face repository and immutable commit SHA."""
    if not MODEL_ID_PATTERN.fullmatch(model_id):
        raise ValueError("invalid Hugging Face model identifier")
    if not REVISION_PATTERN.fullmatch(revision):
        raise ValueError("model revision must be a 40-character commit SHA")


def read_api_token(path: Path) -> str:
    """Read the dispatcher credential only from a private owner-only file."""
    if path.is_symlink() or not path.is_file():
        raise PermissionError("model-download API token path is not a regular file")
    metadata = path.stat()
    mode = stat.S_IMODE(metadata.st_mode)
    owned_by_job = metadata.st_uid == os.getuid() and not mode & 0o077
    protected_group_file = (
        metadata.st_uid == 0
        and metadata.st_gid == os.getgid()
        and mode & 0o040
        and not mode & 0o027
    )
    if not (owned_by_job or protected_group_file):
        raise PermissionError("model-download API token file permissions are unsafe")
    token = path.read_text(encoding="utf-8").strip()
    if not token or len(token.encode("utf-8")) > MAX_API_TOKEN_BYTES:
        raise ValueError("model-download API token file is invalid")
    return token


def parse_credential_response(payload: object) -> str | None:
    """Return only a bounded Hugging Face token from Gail's scoped response."""
    if not isinstance(payload, dict) or payload.get("provider") != "huggingface":
        raise ValueError("Gail returned an invalid provider credential response")
    if payload.get("credential_present") is not True:
        return None
    token = payload.get("access_token")
    if not isinstance(token, str) or not token.strip():
        raise ValueError("Gail marked a provider credential present without a token")
    if len(token.encode("utf-8")) > MAX_PROVIDER_TOKEN_BYTES:
        raise ValueError("provider token exceeds the supported length")
    return token.strip()


def fetch_provider_token(endpoint: str, api_token: str) -> str | None:
    """Fetch the current provider token over TLS with no durable response."""
    parsed_endpoint = urllib.parse.urlsplit(endpoint)
    if (
        parsed_endpoint.scheme != "https"
        or not parsed_endpoint.hostname
        or parsed_endpoint.username is not None
        or parsed_endpoint.password is not None
        or parsed_endpoint.fragment
    ):
        raise ValueError("Gail credential endpoint must be a valid HTTPS URL")
    request = urllib.request.Request(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json",
            "Cache-Control": "no-store",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            if response.status != 200:
                raise SafeDownloaderError(
                    f"Gail model credential request returned HTTP {response.status}"
                )
            final_url = urllib.parse.urlsplit(response.geturl())
            if (
                final_url.scheme != parsed_endpoint.scheme
                or final_url.hostname != parsed_endpoint.hostname
                or final_url.port != parsed_endpoint.port
            ):
                raise RuntimeError("Gail credential endpoint redirected to another origin")
            raw = response.read(16_385)
            if len(raw) > 16_384:
                raise ValueError("Gail model credential response exceeds its size limit")
            return parse_credential_response(json.loads(raw))
    except urllib.error.HTTPError as error:
        # Do not include the exception text: some HTTP client errors can echo
        # request details, while the status code is sufficient for diagnosis.
        raise SafeDownloaderError(
            f"Gail model credential request returned HTTP {error.code}"
        ) from None


def write_provenance(path: Path, record: dict[str, object]) -> None:
    """Atomically save non-secret model identity and commit provenance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o640)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(record, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def download_snapshot(
    model_id: str,
    revision: str,
    cache_root: Path,
    provider_token: str | None,
) -> Path:
    """Fetch and verify one commit-addressed snapshot in the shared cache."""
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=str(cache_root),
            token=provider_token,
            local_files_only=False,
        )
    ).resolve()
    if not snapshot.is_dir() or snapshot.name != revision:
        raise RuntimeError("Hugging Face returned a snapshot with the wrong revision")
    if not any(snapshot.iterdir()):
        raise RuntimeError("Hugging Face returned an empty model snapshot")
    return snapshot


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credential-endpoint", required=True)
    parser.add_argument("--api-token-file", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--provenance-file", required=True, type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> tuple[Path, str]:
    validate_model_reference(args.model_id, args.revision)
    api_token = read_api_token(args.api_token_file)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.cache_root / ".gail-model-download.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        provider_token = fetch_provider_token(args.credential_endpoint, api_token)
        # Keep the provider token in this process only and pass it as a library
        # argument, never through the environment or a child-process argument.
        snapshot = download_snapshot(
            args.model_id,
            args.revision,
            args.cache_root,
            provider_token,
        )
        write_provenance(
            args.provenance_file,
            {
                "schema_version": 1,
                "provider": "huggingface",
                "model_id": args.model_id,
                "requested_revision": args.revision,
                "resolved_revision": args.revision,
                "snapshot_path": str(snapshot),
                "credential_used": provider_token is not None,
                "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
        provider_token = None
        api_token = ""
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return snapshot, args.revision


def main(argv: Sequence[str] | None = None) -> int:
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        # Hugging Face and HTTP libraries may emit request diagnostics. Keep
        # their output out of the durable Slurm log; this helper reports only
        # safe status and exception class names.
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                snapshot, revision = run(args)
        print(f"{snapshot}\t{revision}")
        return 0
    except Exception as error:  # Deliberately omit exception text and traceback.
        print(
            f"Gail model cache preparation failed ({safe_failure_detail(error)}); "
            "sensitive provider output was suppressed.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
