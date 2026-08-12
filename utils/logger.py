"""Console-and-file logging helpers for KKTFormer experiments."""

import atexit
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO


class _TeeStream:
    """Mirror normal output while keeping tqdm refreshes out of log files."""

    def __init__(self, terminal: TextIO, log_file: TextIO):
        self._terminal = terminal
        self._log_file = log_file
        self._lock = threading.Lock()
        self._skip_transient_newline = False

    def write(self, data: str) -> int:
        if not data:
            return 0
        with self._lock:
            self._terminal.write(data)
            self._terminal.flush()

            # tqdm redraws its current line with carriage returns.  Preserve
            # those updates in the terminal, but do not turn every refresh
            # into a separate, unreadable entry in the persistent log.
            if "\r" in data:
                self._skip_transient_newline = True
            elif self._skip_transient_newline and not data.strip("\r\n"):
                self._skip_transient_newline = False
            else:
                self._skip_transient_newline = False
                self._log_file.write(data)
                self._log_file.flush()
        return len(data)

    def flush(self) -> None:
        with self._lock:
            self._terminal.flush()
            self._log_file.flush()

    def isatty(self) -> bool:
        return getattr(self._terminal, "isatty", lambda: False)()

    def fileno(self) -> int:
        return self._terminal.fileno()


def setup_logger(log_dir: str = "./logs", name: Optional[str] = None) -> Path:
    """Mirror stdout/stderr to a timestamped log file and return its path."""

    output_dir = Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = name or "run"
    log_path = output_dir / f"{prefix}_{timestamp}_pid{os.getpid()}.log"
    log_file = log_path.open("a", encoding="utf-8", buffering=1)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tee_stdout = _TeeStream(original_stdout, log_file)
    tee_stderr = _TeeStream(original_stderr, log_file)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    def close_log() -> None:
        # Restore the process streams before closing the file.  Wrappers such
        # as ``conda run`` may emit output after application atexit handlers.
        if sys.stdout is tee_stdout:
            sys.stdout = original_stdout
        if sys.stderr is tee_stderr:
            sys.stderr = original_stderr
        log_file.flush()
        log_file.close()

    atexit.register(close_log)
    return log_path.resolve()
