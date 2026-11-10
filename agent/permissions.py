class PermissionDenied(Exception):
    """Operation is outside the configured scope or lacks trusted confirmation."""


def require_confirmation(action: str, confirmations: set[str]):
    if action not in confirmations:
        raise PermissionDenied(f"Explicit confirmation required: {action}")


def require_namespace(requested: str, approved: str):
    if requested != approved:
        raise PermissionDenied("Namespace is outside the approved scope")
