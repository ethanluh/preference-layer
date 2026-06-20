"""Polite-crawl primitives: token-bucket rate limiting + robots.txt respect.

The kickoff (``docs/phase1-kickoff.md`` B1) requires the iFixit/Notebookcheck
crawl to be polite and to respect ``robots.txt``, and the Reddit connector to be
rate-limited. Both connectors share these primitives.

Kept dependency-free and injectable: the :class:`RateLimiter` takes a clock and
sleep function so tests can drive it deterministically without wall-clock waits,
and :class:`RobotsPolicy` parses a robots.txt body in-memory (no network).
"""

from __future__ import annotations

import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class RateLimiter:
    """Token bucket: at most ``rate`` requests/second, with a small burst.

    ``clock``/``sleep`` are injectable for tests. ``acquire`` blocks (via
    ``sleep``) only when the bucket is empty, so steady-state throughput is
    capped without busy-waiting.
    """

    rate: float                  # sustained requests per second
    burst: float = 1.0           # bucket capacity
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    _tokens: float = 0.0
    _last: float | None = None

    def acquire(self) -> None:
        now = self.clock()
        if self._last is None:
            self._tokens = self.burst
            self._last = now
        # Refill proportional to elapsed time.
        self._tokens = min(self.burst, self._tokens + (now - self._last) * self.rate)
        self._last = now
        if self._tokens < 1.0:
            wait = (1.0 - self._tokens) / self.rate
            self.sleep(wait)
            self._tokens = 0.0
            self._last = self.clock()
        else:
            self._tokens -= 1.0


class RobotsPolicy:
    """Minimal robots.txt evaluator (User-agent / Disallow / Allow / Crawl-delay).

    Supports the directives crawlers actually need for a polite single-agent
    crawl. ``can_fetch(url)`` checks the longest-matching rule (Allow wins ties),
    matching the de-facto robots.txt precedence. Not a full RFC 9309 parser --
    it deliberately covers the common subset and errs toward *not* fetching when
    a path is disallowed.
    """

    def __init__(self, body: str, user_agent: str):
        self.user_agent = user_agent
        self.crawl_delay: float | None = None
        self._rules: list[tuple[str, bool]] = []  # (path_prefix, allowed)
        self._parse(body)

    def _parse(self, body: str) -> None:
        # Collect rule groups; apply the group matching our UA, else '*'.
        groups: dict[str, list[tuple[str, bool]]] = {}
        delays: dict[str, float] = {}
        current_agents: list[str] = []
        for raw_line in body.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            field, _, value = line.partition(":")
            field = field.strip().lower()
            value = value.strip()
            if field == "user-agent":
                current_agents = [value.lower()]
                groups.setdefault(value.lower(), [])
            elif field in ("disallow", "allow") and current_agents:
                allowed = field == "allow"
                for agent in current_agents:
                    # An empty Disallow means "allow all" -> no rule.
                    if field == "disallow" and value == "":
                        continue
                    groups.setdefault(agent, []).append((value, allowed))
            elif field == "crawl-delay" and current_agents:
                try:
                    for agent in current_agents:
                        delays[agent] = float(value)
                except ValueError:
                    pass

        ua = self.user_agent.lower()
        chosen = ua if ua in groups else "*"
        self._rules = groups.get(chosen, [])
        self.crawl_delay = delays.get(chosen, delays.get("*"))

    def can_fetch(self, url: str) -> bool:
        path = urllib.parse.urlsplit(url).path or "/"
        best_len = -1
        decision = True  # default allow when no rule matches
        for prefix, allowed in self._rules:
            if path.startswith(prefix) and len(prefix) > best_len:
                best_len = len(prefix)
                decision = allowed
            elif path.startswith(prefix) and len(prefix) == best_len and allowed:
                decision = True  # Allow wins ties
        return decision
