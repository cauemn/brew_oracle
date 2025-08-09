def Field(default=None, **kwargs):
    return default


def field_validator(*args, **kwargs):
    def decorator(func):
        return func
    return decorator
