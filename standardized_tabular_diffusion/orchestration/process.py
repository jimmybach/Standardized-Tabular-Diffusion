"""Isolated subprocess execution with explicit resource and failure boundaries."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Mapping, Sequence

_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|secret|credential)\b"
    r"(\s*[:=]\s*)([^\s,;]+)"
)
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+\-/]+=*")
_GITHUB_TOKEN = re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{12,}\b")
_QUOTED_ABSOLUTE_PATH = re.compile(r"(?P<quote>[\"'])(?:[A-Za-z]:[\\/]|/(?!/))[^\"']+(?P=quote)")


@dataclass(frozen=True)
class ProcessLimits:
    timeout_seconds: float | None = None
    memory_limit_bytes: int | None = None
    poll_interval_seconds: float = 0.1

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.memory_limit_bytes is not None and self.memory_limit_bytes <= 0:
            raise ValueError("memory_limit_bytes must be positive")
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")


@dataclass(frozen=True)
class ProcessOutcome:
    exit_code: int | None
    wall_seconds: float
    cpu_seconds: float | None
    peak_rss_bytes: int | None
    peak_accelerator_memory_bytes: int | None
    timed_out: bool
    memory_limit_exceeded: bool
    interrupted: bool
    launch_error: str | None
    reliability: tuple[str, ...]
    diagnostic_tail: tuple[str, ...]


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def redact_text(value: str, *, path_aliases: Mapping[str, str] | None = None) -> str:
    """Redact common credential forms and caller-declared unsafe path prefixes."""

    redacted = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}{match.group(2)}<redacted>", value)
    redacted = _BEARER.sub("Bearer <redacted>", redacted)
    redacted = _GITHUB_TOKEN.sub("<redacted-github-token>", redacted)
    aliases = dict(path_aliases or {})
    for path, alias in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if path:
            redacted = redacted.replace(path, alias).replace(path.replace("\\", "/"), alias)
    redacted = _QUOTED_ABSOLUTE_PATH.sub(lambda match: f"{match.group('quote')}<host-path>{match.group('quote')}", redacted)
    return redacted


class _EventLog:
    def __init__(self, path: Path, *, stage_id: str, path_aliases: Mapping[str, str]) -> None:
        if path.exists() and path.is_symlink():
            raise ValueError(f"Refusing to write a symlinked log: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8", newline="\n")
        self._stage_id = stage_id
        self._path_aliases = path_aliases
        self._lock = threading.Lock()
        self.tail: deque[str] = deque(maxlen=50)

    def write(self, event: str, *, stream: str | None = None, message: str | None = None, **extra: Any) -> None:
        payload: dict[str, Any] = {
            "event_schema_version": "1.0.0",
            "timestamp": _utc_timestamp(),
            "stage_id": self._stage_id,
            "event": event,
        }
        if stream is not None:
            payload["stream"] = stream
        if message is not None:
            safe = redact_text(message.rstrip("\r\n"), path_aliases=self._path_aliases)
            payload["message"] = safe
            self.tail.append(safe)
        payload.update(extra)
        with self._lock:
            self._handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            self._handle.close()


def _drain(stream: IO[str], name: str, events: _EventLog) -> None:
    try:
        for line in iter(stream.readline, ""):
            events.write("process-output", stream=name, message=line)
    finally:
        stream.close()


def _process_tree_usage(pid: int) -> tuple[int | None, float | None, set[int], str | None]:
    try:
        import psutil
    except ImportError:
        return None, None, {pid}, "psutil_unavailable"
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None, None, {pid}, None
    rss = 0
    cpu = 0.0
    pids: set[int] = set()
    observed = False
    for process in processes:
        try:
            pids.add(process.pid)
            rss += int(process.memory_info().rss)
            times = process.cpu_times()
            cpu += float(times.user + times.system)
            observed = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return (rss if observed else None), (cpu if observed else None), pids or {pid}, None


def _accelerator_process_memory(pids: set[int]) -> tuple[int | None, str | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None, "accelerator_memory_sampling_unavailable"
    total_mib = 0.0
    unsupported = False
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            process_id = int(fields[0])
            memory_mib = float(fields[1])
        except ValueError:
            unsupported = True
            continue
        if process_id in pids:
            total_mib += memory_mib
    if unsupported:
        return None, "accelerator_memory_sampling_unavailable"
    return int(total_mib * 1024 * 1024), None


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    try:
        import psutil

        root = psutil.Process(process.pid)
        descendants = root.children(recursive=True)
        for child in reversed(descendants):
            try:
                child.terminate()
            except psutil.Error:
                pass
        try:
            root.terminate()
        except psutil.Error:
            pass
        _, alive = psutil.wait_procs([*descendants, root], timeout=2)
        for item in alive:
            try:
                item.kill()
            except psutil.Error:
                pass
    except Exception:  # process termination is best effort after a boundary fires
        try:
            if os.name == "posix":
                getattr(os, "killpg")(process.pid, getattr(signal, "SIGKILL"))
            else:
                process.kill()
        except OSError:
            pass


def run_isolated_process(
    command: Sequence[str],
    *,
    cwd: str | Path,
    log_path: str | Path,
    stage_id: str,
    limits: ProcessLimits,
    environment: Mapping[str, str] | None = None,
    path_aliases: Mapping[str, str] | None = None,
) -> ProcessOutcome:
    """Run one stage in a process group and preserve only redacted structured logs."""

    if not command or any(not isinstance(item, str) or not item for item in command):
        raise ValueError("command must contain non-empty strings")
    if limits.memory_limit_bytes is not None:
        try:
            import psutil  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("A memory limit requires the orchestration dependency psutil") from exc
    aliases = dict(path_aliases or {})
    aliases.setdefault(str(Path.home()), "<user-home>")
    events = _EventLog(Path(log_path), stage_id=stage_id, path_aliases=aliases)
    started = time.perf_counter()
    reliability: set[str] = set()
    launch_error: str | None = None
    try:
        popen_kwargs: dict[str, Any] = {}
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            process = subprocess.Popen(
                list(command),
                cwd=Path(cwd),
                env=None if environment is None else dict(environment),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **popen_kwargs,
            )
        except OSError as exc:
            launch_error = redact_text(str(exc), path_aliases=aliases)
            events.write("process-launch-failed", message=launch_error)
            return ProcessOutcome(
                exit_code=None,
                wall_seconds=time.perf_counter() - started,
                cpu_seconds=None,
                peak_rss_bytes=None,
                peak_accelerator_memory_bytes=None,
                timed_out=False,
                memory_limit_exceeded=False,
                interrupted=False,
                launch_error=launch_error,
                reliability=("process_not_started",),
                diagnostic_tail=tuple(events.tail),
            )
        events.write("process-started")
        assert process.stdout is not None and process.stderr is not None
        threads = [
            threading.Thread(target=_drain, args=(process.stdout, "stdout", events), daemon=True),
            threading.Thread(target=_drain, args=(process.stderr, "stderr", events), daemon=True),
        ]
        for thread in threads:
            thread.start()
        peak_rss: int | None = None
        peak_accelerator: int | None = None
        cpu_seconds: float | None = None
        timed_out = False
        memory_exceeded = False
        interrupted = False
        next_accelerator_sample = 0.0
        try:
            while process.poll() is None:
                elapsed = time.perf_counter() - started
                if limits.timeout_seconds is not None and elapsed > limits.timeout_seconds:
                    timed_out = True
                    events.write("process-boundary-fired", boundary="timeout")
                    _terminate_process_tree(process)
                    break
                rss, cpu, pids, usage_warning = _process_tree_usage(process.pid)
                if usage_warning:
                    reliability.add(usage_warning)
                if rss is not None:
                    peak_rss = rss if peak_rss is None else max(peak_rss, rss)
                if cpu is not None:
                    cpu_seconds = cpu if cpu_seconds is None else max(cpu_seconds, cpu)
                if limits.memory_limit_bytes is not None and rss is not None and rss > limits.memory_limit_bytes:
                    memory_exceeded = True
                    events.write("process-boundary-fired", boundary="memory")
                    _terminate_process_tree(process)
                    break
                if elapsed >= next_accelerator_sample:
                    accelerator, accelerator_warning = _accelerator_process_memory(pids)
                    if accelerator_warning:
                        reliability.add(accelerator_warning)
                    if accelerator is not None:
                        peak_accelerator = (
                            accelerator if peak_accelerator is None else max(peak_accelerator, accelerator)
                        )
                    next_accelerator_sample = elapsed + 0.5
                time.sleep(limits.poll_interval_seconds)
        except KeyboardInterrupt:
            interrupted = True
            events.write("process-boundary-fired", boundary="interruption")
            _terminate_process_tree(process)
        try:
            exit_code = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _terminate_process_tree(process)
            exit_code = process.wait(timeout=5)
        except KeyboardInterrupt:
            interrupted = True
            events.write("process-boundary-fired", boundary="interruption")
            _terminate_process_tree(process)
            exit_code = process.wait(timeout=5)
        for thread in threads:
            thread.join(timeout=5)
        rss, cpu, _pids, usage_warning = _process_tree_usage(process.pid)
        if usage_warning:
            reliability.add(usage_warning)
        if rss is not None:
            peak_rss = rss if peak_rss is None else max(peak_rss, rss)
        if cpu is not None:
            cpu_seconds = cpu if cpu_seconds is None else max(cpu_seconds, cpu)
        events.write("process-ended", exit_code=exit_code)
        return ProcessOutcome(
            exit_code=exit_code,
            wall_seconds=time.perf_counter() - started,
            cpu_seconds=cpu_seconds,
            peak_rss_bytes=peak_rss,
            peak_accelerator_memory_bytes=peak_accelerator,
            timed_out=timed_out,
            memory_limit_exceeded=memory_exceeded,
            interrupted=interrupted,
            launch_error=launch_error,
            reliability=tuple(sorted(reliability)),
            diagnostic_tail=tuple(events.tail),
        )
    finally:
        events.close()
