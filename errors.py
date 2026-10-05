"""Application error type shared by the service layer and the web layer."""


class ApiError(Exception):
    """An error that should be reported to the user with a clear message."""

    def __init__(self, status_code, code, message, extra=None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.extra = extra or {}

    def to_detail(self):
        detail = {"code": self.code, "message": self.message}
        detail.update(self.extra)
        return detail
