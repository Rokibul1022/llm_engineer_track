"""Exceptions for the extraction service."""

class ExtractionFailure(Exception):
    """Raised when an extraction fails structural validation or exhausting retry budget."""
    pass
