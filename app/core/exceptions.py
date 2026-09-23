"""Custom exceptions for runtime interceptor and multi-agent pipelines."""


class InterceptorBaseException(Exception):
    """Base exception for all interceptor domain errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class LLMProviderError(InterceptorBaseException):
    """Raised when an LLM provider encounters an API, timeout, or parsing error."""

    pass


class LLMUnavailableError(LLMProviderError):
    """Raised when an external LLM provider or model server is not reachable."""

    pass


class SchemaValidationError(InterceptorBaseException):
    """Raised when structured LLM output violates schema requirements."""

    pass


class CalibrationError(InterceptorBaseException):
    """Raised when model calibration fails or calibration profile is missing."""

    pass


class ExperimentError(InterceptorBaseException):
    """Raised when benchmark or ablation experiment execution fails."""

    pass


class RepairExhaustedError(InterceptorBaseException):
    """Raised when maximum repair attempts have been exhausted."""

    pass
