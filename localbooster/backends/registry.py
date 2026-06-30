"""Backend factory helpers."""

from __future__ import annotations

from localbooster.config import BackendName


def load_backend(name: BackendName, model_id: str, **kwargs: object):
    """Load a backend lazily so optional dependencies remain optional."""

    if name == "transformers":
        from localbooster.backends.transformers_backend import TransformersBackend

        return TransformersBackend(model_id, **kwargs)
    if name == "mlx":
        from localbooster.backends.mlx_backend import MLXBackend

        return MLXBackend(model_id, **kwargs)
    raise ValueError(f"unknown backend {name!r}")

