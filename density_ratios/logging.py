import logging

import rich.logging as rl

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[rl.RichHandler(rich_tracebacks=True, tracebacks_show_locals=True)],
)


def get_logger(name):
    return logging.getLogger(name)
