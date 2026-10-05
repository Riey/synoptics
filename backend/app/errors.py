"""The API-level failure type shared by every route module.

It lives outside ``main`` so a route module (for example the tracker transport) can raise it without
importing the application module that wires the routes together.
"""

from __future__ import annotations

from typing import Any

from backend.app.provider import InvalidOutputStage, ProviderError, UnavailableStage


class StageError(ProviderError):
    """A local-endpoint failure (tracker, local follower): distinct from the paid provider's own errors.

    It extends :class:`ProviderError` so the existing API error mapping keeps applying.
    """


class StageConfigError(StageError):
    """A local endpoint is selected but its configuration is missing or unusable. Fail closed."""


class ApiFailure(Exception):
    """An API-level failure.

    ``detail`` is optional and used where a fail-closed configuration error must name the prerequisite that
    is missing (an environment variable name, never a value).

    ``reason`` is a closed-set provider stage token, never free text: it exists so a rejected provider
    answer is attributable from the response alone (which validator refused it, or whether the provider
    never answered). It is typed as the stage enums, so prose cannot be attached to it by accident.

    ``extra`` carries the few structured fields a frozen error body requires (an active run id, the frame
    sequence a client is expected to continue from). It is merged into the error body verbatim; it never
    carries prose, and never a track state.
    """

    def __init__(
        self,
        status: int,
        code: str,
        retry_after: int | str | None = None,
        detail: str | None = None,
        reason: InvalidOutputStage | UnavailableStage | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.status = status
        self.code = code
        self.retry_after = retry_after
        self.detail = detail
        self.reason = reason
        self.extra = extra
