"""Backend-specific exceptions."""


class OptionalDependencyError(RuntimeError):
    """Raised when a backend optional dependency is not installed."""

