"""HTTP transport for the PTP credential store (PTP spec §4).

A thin FastAPI adapter over :class:`~preferencelayer.ptp.store.CredentialStore`.
The store's logic — selective disclosure, the DP update + re-sign, scope checks —
is reused verbatim; this layer only maps HTTP requests/responses and enforces the
bearer-token boundary. Import :func:`build_app` lazily so the core package stays
usable without the optional ``http`` extra installed.
"""

from .app import build_app

__all__ = ["build_app"]
