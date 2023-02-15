__all__ = []


def is_iterable(obj) -> bool:
    try:
        iter(obj)
        return True
    except TypeError:
        return False


class OutsideOfContextError(Exception):
    pass


def raise_if_outside_context(method):
    """Raise an exception if the method is called outside of a context manager"""

    def decorator(self, *args, **kwargs):
        if not self._inside_context:
            raise OutsideOfContextError(
                "This method should be called inside a context manager"
            )
        return method(self, *args, **kwargs)

    return decorator


def raise_if_outside_write_context(method):
    """Raise an exception if the method is called outside of a write context manager"""

    def decorator(self, *args, **kwargs):
        if not self._inside_context and self._mode != "r+b":
            raise OutsideOfContextError(
                "This method should be called inside a write context manager"
            )
        return method(self, *args, **kwargs)

    return decorator


def provide_context_if_needed(method):
    """If the method is called outside of a context manager, provide one"""

    def decorator(self, *args, **kwargs):
        if not self._inside_context:
            with self:
                return method(self, *args, **kwargs)
        return method(self, *args, **kwargs)

    return decorator
