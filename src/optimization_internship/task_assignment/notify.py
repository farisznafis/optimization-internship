"""Progress messages to the log and, optionally, a Discord webhook."""

from __future__ import annotations

import datetime
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

import requests

log = logging.getLogger(__name__)


class Notifier:
    """Log a message and, if a webhook URL is set, post it to Discord.

    A failing webhook is logged as a warning and never interrupts the solve.
    """

    def __init__(self, webhook_url: str | None = None) -> None:
        self.webhook_url = webhook_url

    def __call__(self, message: str) -> None:
        log.info(message)
        if self.webhook_url:
            self._send_discord(message)

    def _send_discord(self, message: str) -> None:
        timestamp = datetime.datetime.now().strftime("(%H.%M %d/%m/%Y)")
        try:
            response = requests.post(self.webhook_url, json={"content": f"{timestamp} {message}"}, timeout=10)
            response.raise_for_status()
        except requests.RequestException as error:
            log.warning("Discord notification failed: %s", error)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Report start, duration and failure of a block of work."""
        self(f"{name} START")
        start = time.perf_counter()
        try:
            yield
        except Exception as error:
            self(f"{name} failed: {error}")
            raise
        self(f"{name} finished in {time.perf_counter() - start:.1f} s")
