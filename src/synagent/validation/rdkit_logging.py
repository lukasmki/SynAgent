import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager

from rdkit import rdBase

# Route RDKit's error and warning logs through Python logging so the tools can
# capture them. Outside a capture they still print to stderr as before.
rdBase.LogToPythonLogger()
_rdkit_logger = logging.getLogger("rdkit")
_TIMESTAMP = re.compile(r"^\[\d{2}:\d{2}:\d{2}\] ")


class LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(_TIMESTAMP.sub("", record.getMessage()).rstrip())

    def annotate(self, message: str) -> str:
        """Appends the captured RDKit messages, without repeats, to `message`."""
        if not self.lines:
            return message
        return message + "\nRDKit log:\n" + "\n".join(dict.fromkeys(self.lines))


@contextmanager
def capture_rdkit_log() -> Iterator[LogCapture]:
    """Collects RDKit log messages instead of printing them."""
    capture = LogCapture()
    handlers = _rdkit_logger.handlers
    _rdkit_logger.handlers = [capture]
    try:
        yield capture
    finally:
        _rdkit_logger.handlers = handlers
