"""PreferenceLayer: portable preference graph infrastructure (Phase 0 prototype)."""

__version__ = "0.1.0"

from . import attributes

__all__ = ["attributes", "__version__"]


def __getattr__(name):  # lazy submodule access without importing optional deps eagerly
    if name == "qil":
        from . import qil

        return qil
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
