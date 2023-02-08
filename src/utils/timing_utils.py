import logging
import time

import contextlib
from datetime import timedelta
import os
import errno
import signal

logger = logging.getLogger(__name__)

# Misc logger setup so a debug log statement gets printed on stdout.
logger.setLevel("DEBUG")
logger.propagate = False
handler = logging.StreamHandler()
log_format = "%(asctime)s %(levelname)s -- %(message)s"
formatter = logging.Formatter(log_format)
handler.setFormatter(formatter)
logger.addHandler(handler)


@contextlib.contextmanager
def time_measure(ident, _logger=logger, show_started=True):
    if show_started:
        _logger.info("%s Started" % ident)
    start_time = time.time()
    yield
    elapsed_time = str(timedelta(seconds=time.time() - start_time))
    _logger.info("%s Finished in %s " % (ident, elapsed_time))


@contextlib.contextmanager
def record_elapsed_time(time_sequence):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    time_sequence.append(elapsed_time)


DEFAULT_TIMEOUT_MESSAGE = os.strerror(errno.ETIME)


class timeout(contextlib.ContextDecorator):
    def __init__(self, seconds, *, timeout_message=DEFAULT_TIMEOUT_MESSAGE, suppress_timeout_errors=False):
        self.seconds = int(seconds)
        self.timeout_message = timeout_message
        self.suppress = bool(suppress_timeout_errors)

    def _timeout_handler(self, signum, frame):
        raise TimeoutError(self.timeout_message)

    def __enter__(self):
        if self.seconds > 0:
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)
        if self.suppress and exc_type is TimeoutError:
            return True
