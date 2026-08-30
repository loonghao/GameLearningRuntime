"""Exception hierarchy for Game Learning Runtime."""


class GLRError(Exception):
    """Base class for all GLR errors."""


class ContractViolation(GLRError, ValueError):
    """Raised when an environment violates its declared contract."""


class OptionalDependencyError(GLRError, ImportError):
    """Raised when an optional integration is used without its dependencies."""
