import logging


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return a shared logger for the project.

    Parameters
    ----------
    level : int, optional
        Logging level, by default ``logging.INFO``.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger("brew_oracle")
    if not logger.handlers:
        logging.basicConfig(level=level)
    logger.setLevel(level)
    return logger
