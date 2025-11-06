import logging
from .state import PartialState


class MultiProcessAdapter(logging.LoggerAdapter):
    """
    An adapter to assist with logging in multiprocess.

    `log` takes in an additional `main_process_only` kwarg, which dictates whether it should be called on all processes
    or only the main executed one. Default is `main_process_only=True`.

    Does not require an `Accelerator` object to be created first.
    """

    @staticmethod
    def _should_log(main_process_only):
        "Check if log should be performed"
        state = PartialState()
        return not main_process_only or (main_process_only and state.is_main_process)

    def log(self, level, msg, *args, **kwargs):
        """
        Delegates logger call after checking if we should log.

        Accepts a new kwarg of `main_process_only`, which will dictate whether it will be logged across all processes
        or only the main executed one. Default is `True` if not passed
        """
        main_process_only = kwargs.pop("main_process_only", True)
        if self.isEnabledFor(level) and self._should_log(main_process_only):
            msg, kwargs = self.process(msg, kwargs)
            self.logger.log(level, msg, *args, **kwargs)


def get_logger(name: str, log_level: str = "INFO"):
    """
    Returns a `logging.Logger` for `name` that can handle multiprocessing.

    If a log should be called on all processes, pass `main_process_only=False`

    Args:
        name (`str`):
            The name for the logger, such as `__file__`
        log_level (`str`, *optional*):
            The log level to use. If not passed, will default to the `LOG_LEVEL` environment variable, or `INFO` if not

    Example:

    ```python
    >>> from accelerate.logging import get_logger

    >>> logger = get_logger(__name__)

    >>> logger.info("My log", main_process_only=False)
    >>> logger.debug("My log", main_process_only=True)

    >>> logger = get_logger(__name__, log_level="DEBUG")
    >>> logger.info("My log")
    >>> logger.debug("My second log")
    ```
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level.upper())
    return MultiProcessAdapter(logger, {})
