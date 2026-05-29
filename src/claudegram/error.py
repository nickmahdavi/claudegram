from enum import StrEnum

import anthropic

class ErrorClass(StrEnum):
    TRANSIENT = "transient error"
    CREDIT = "credit balance exhausted"
    AUTH = "authentication error"
    MODEL_NOT_FOUND = "model not found error"
    BAD_REQUEST = "bad request"
    UNKNOWN = "unknown error"


def classify_error(exc: BaseException) -> tuple[ErrorClass, str]:
    if isinstance(exc, anthropic.APIStatusError):
        code = exc.status_code
        if code == 402:
            return ErrorClass.CREDIT, "credit balance exhausted"
        if code == 529:
            return ErrorClass.TRANSIENT, "anthropic overloaded (529)"
        if code == 504:
            return ErrorClass.TRANSIENT, "anthropic timeout (504)"
        if code == 503:
            return ErrorClass.TRANSIENT, "anthropic unavailable (503)"
    if isinstance(exc, anthropic.RateLimitError):
        return ErrorClass.TRANSIENT, "rate limited (429)"
    if isinstance(exc, anthropic.InternalServerError):
        return ErrorClass.TRANSIENT, "anthropic 5xx"
    if isinstance(exc, anthropic.APIConnectionError):
        return ErrorClass.TRANSIENT, "connection failure"
    if isinstance(exc, anthropic.AuthenticationError):
        return ErrorClass.AUTH, "auth failure (bad/expired key)"
    if isinstance(exc, anthropic.PermissionDeniedError):
        return ErrorClass.AUTH, "permission denied"
    if isinstance(exc, anthropic.NotFoundError):
        return ErrorClass.MODEL_NOT_FOUND, "model not found"
    if isinstance(exc, anthropic.BadRequestError):
        # Fallback for legacy/edge 400s tagged as billing. 402 is canonical
        msg = str(exc).lower()
        if any(k in msg for k in ("credit balance", "billing", "quota", "your credit")):
            return ErrorClass.CREDIT, "credit balance exhausted"
        return ErrorClass.BAD_REQUEST, f"bad request: {str(exc)[:120]}"
    if isinstance(exc, anthropic.UnprocessableEntityError):

        return ErrorClass.BAD_REQUEST, "unprocessable entity"
    if isinstance(exc, anthropic.APIError):
        return ErrorClass.UNKNOWN, f"anthropic api error: {type(exc).__name__}"
    return ErrorClass.UNKNOWN, type(exc).__name__


USER_ERROR_REPLIES: dict[ErrorClass, str] = {
    ErrorClass.TRANSIENT: "Temporary issue talking to the API. Try again in a sec.",
    ErrorClass.CREDIT: "Out of API credits.",
    ErrorClass.AUTH: "API auth is broken.",
    ErrorClass.MODEL_NOT_FOUND: "That model isn't available. Try `/model` to see supported options.",
    ErrorClass.BAD_REQUEST: "Bad API request.",
    ErrorClass.UNKNOWN: "Unexpected error.",
}

ADMIN_FAILURE_DMS: dict[ErrorClass, str] = {
    ErrorClass.CREDIT: "{count} consecutive credit API errors in chat `{chat_id}`",
    ErrorClass.AUTH: "{count} auth errors in chat `{chat_id}` ({desc})",
    ErrorClass.TRANSIENT: (
        "{count} transient API errors in chat `{chat_id}` ({desc})"
    ),
    ErrorClass.MODEL_NOT_FOUND: (
        "{count} model-not-found errors in chat `{chat_id}` ({desc})"
    ),
    ErrorClass.BAD_REQUEST: (
        "{count} bad-request errors in chat `{chat_id}` ({desc})."
    ),
    ErrorClass.UNKNOWN: (
        "{count} unrecognized errors in chat `{chat_id}` ({desc})."
    ),
}

ADMIN_RECOVERY_DM = "back online in chat `{chat_id}`, recovered after {count} failures"


def user_reply(cls: ErrorClass) -> str:
    """User-facing reply text for an error class, with UNKNOWN as the fallback."""
    return USER_ERROR_REPLIES.get(cls, USER_ERROR_REPLIES[ErrorClass.UNKNOWN])


# ---- credential-related user-facing messages ---------------------------
# (kept parse_mode-free at the call sites, per the bot's no-markdown-on-
# user-controlled-content convention; these are plain text.)

def no_credential_reply(is_private: bool) -> str:
    """Polite refusal when a request resolves to no usable credential."""
    if is_private:
        return (
            "I can't answer without credentials to bill this to. Send /setkey "
            "<your-anthropic-key> here in this DM (I'll delete the message), or "
            "ask an admin to add you to the shared pool."
        )
    return (
        "I can't answer here without credentials to bill this to. DM me /setkey "
        "<your-anthropic-key>, or ask an admin to add you to the shared pool."
    )


def credential_broken_reply() -> str:
    """When we couldn't even build a client from a stored credential."""
    return (
        "Your stored credential couldn't be used. Try /forgetkey then /setkey "
        "again, or check /keystatus."
    )


# User-owned credential failures, attributed to the user (never the bot/pool).
_USER_FAILURE_REPLIES: dict[ErrorClass, str] = {
    ErrorClass.AUTH: "Your API key was rejected (invalid or expired). Re-add it with /setkey; see /keystatus.",
    ErrorClass.CREDIT: "Your account is out of API credits. Top up, then try again.",
    # TRANSIENT covers a 429 on the user's OWN account (their rate limit) as well
    # as Anthropic-wide blips — so don't promise it's "just temporary".
    ErrorClass.TRANSIENT: "Temporary problem on your credential — possibly your account's rate limit. Try again shortly.",
}


def user_credential_failed_reply(err_class: ErrorClass) -> str:
    """Reply for a failure on the *user's own* credential. AUTH/CREDIT get
    credential-specific wording; everything else falls back to the generic
    per-class reply (transient, etc.)."""
    return _USER_FAILURE_REPLIES.get(err_class, user_reply(err_class))


def admin_failure_dm(cls: ErrorClass, chat_id: int, count: int, desc: str) -> str:
    """Render the admin DM body for a failure streak. UNKNOWN is the fallback template."""
    template = ADMIN_FAILURE_DMS.get(cls, ADMIN_FAILURE_DMS[ErrorClass.UNKNOWN])
    return template.format(chat_id=chat_id, count=count, desc=desc)


def admin_recovery_dm(chat_id: int, count: int) -> str:
    """Render the admin DM body announcing recovery from a failure streak."""
    return ADMIN_RECOVERY_DM.format(chat_id=chat_id, count=count)
