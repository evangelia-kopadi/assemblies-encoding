"""Shared causal-discovery exceptions."""


class CausalDiscoveryError(RuntimeError):
    """Base class for causal-discovery failures."""


class CausalDiscoveryBackendError(CausalDiscoveryError):
    """Raised when a causal-discovery backend fails to produce a graph."""
