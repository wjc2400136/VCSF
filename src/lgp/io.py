from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=".{}.".format(path.name),
            suffix=".part",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
        temporary_name = None
        if hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    # Serialize before opening the destination so a rejected NaN/Inf value
    # cannot create or partially extend an otherwise valid evidence stream.
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized + "\n")


def file_digest(path: Path, algorithm: str = "sha256", chunk_size: int = 8 << 20) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def iter_download(
    url: str,
    destination: Path,
    expected_digest: Optional[str] = None,
    digest_algorithm: str = "md5",
    chunk_size: int = 8 << 20,
    attempts: int = 5,
) -> Iterator[Dict[str, Any]]:
    """Download with retries and a resumable .part file.

    Direct HTTPS is attempted before inherited proxy settings. This avoids a
    stale system proxy while still retaining a proxy fallback for restricted
    networks.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Optional[Exception] = None
    completed = False
    for attempt in range(1, attempts + 1):
        offset = temporary.stat().st_size if temporary.exists() else 0
        headers = {"Range": "bytes={}-".format(offset)} if offset else {}
        session = requests.Session()
        # First prefer a clean direct connection; on the final attempt retain a
        # chance to use an explicitly configured corporate/system proxy.
        session.trust_env = attempt == attempts
        retries = Retry(
            total=2,
            connect=2,
            read=2,
            backoff_factor=0.75,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(("GET",)),
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))
        try:
            with session.get(
                url, stream=True, timeout=(20, 180), headers=headers
            ) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    offset = 0
                    temporary.unlink(missing_ok=True)
                mode = "ab" if offset else "wb"
                total = response.headers.get("Content-Range") or response.headers.get(
                    "Content-Length"
                )
                with temporary.open(mode) as handle:
                    downloaded = offset
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
                        yield {
                            "downloaded": downloaded,
                            "reported_total": total,
                            "url": url,
                            "attempt": attempt,
                        }
            completed = True
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
        finally:
            session.close()
    if not completed:
        raise RuntimeError(
            "Download failed after {} attempts (partial file retained for resume): {}".format(
                attempts, temporary
            )
        ) from last_error
    os.replace(str(temporary), str(destination))
    if expected_digest:
        actual = file_digest(destination, digest_algorithm)
        if actual.lower() != expected_digest.lower():
            raise RuntimeError(
                "{} mismatch for {}: expected {}, got {}".format(
                    digest_algorithm.upper(), destination, expected_digest, actual
                )
            )
    yield {"downloaded": destination.stat().st_size, "complete": True, "path": str(destination)}
