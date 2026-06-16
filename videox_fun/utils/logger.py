import logging
import os
import sys
import torch.distributed as dist


old_factory = logging.getLogRecordFactory()

def record_factory(*args, **kwargs):
    record = old_factory(*args, **kwargs)
    record.relpath = os.path.relpath(record.pathname, start=os.getcwd())  # Relative to the current working directory.
    return record

logging.setLogRecordFactory(record_factory)

# Read LOG_LEVEL from the environment, defaulting to INFO.
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
LOG_LEVEL_MAP = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

# Resolve the requested logging level.
level = LOG_LEVEL_MAP.get(LOG_LEVEL, logging.INFO)

if os.environ.get("RANK") != "0":
    level = logging.WARNING

# Reconfigure logging with force=True so LOG_LEVEL always takes effect.
logging.basicConfig(
    level=level, 
    format='%(asctime)s - %(relpath)s:%(lineno)d - %(levelname)s: %(message)s',
    force=True  # Override any previous logging configuration.
)

logger = logging.getLogger(__name__)
print(f"Log level set to: {LOG_LEVEL} ({level})")
print("Log handlers: ", logger.handlers)


class LoggerWriter:
    def __init__(self, level_func):
        self.level_func = level_func
        self._buffer = ""

    def write(self, message):
        message = message.rstrip()
        rank = None if not dist.is_initialized() else dist.get_rank()
        if message:
            if rank is not None:
                message = f"[rank={rank}] {message}"
            self.level_func(message, stacklevel=2)

    def flush(self):
        pass  # Keep compatibility with file-like interfaces.
