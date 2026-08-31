"""Exception hierarchy for Game Learning Runtime."""


class GLRError(Exception):
    """Base class for all GLR errors."""


class ContractViolation(GLRError, ValueError):
    """Raised when an environment violates its declared contract."""


class OptionalDependencyError(GLRError, ImportError):
    """Raised when an optional integration is used without its dependencies."""


class HostProtocolError(GLRError, ValueError):
    """Raised when a Runtime Host violates framing or wire-schema rules."""


class HostRemoteError(GLRError):
    """A structured, non-retried error returned by a Runtime Host."""

    def __init__(self, *, code: str, message: str, retryable: bool) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.retryable = retryable
