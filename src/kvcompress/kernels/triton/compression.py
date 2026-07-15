"""Triton kernel shim. Implemented in M8."""

raise NotImplementedError("M8: triton compression kernel is implemented in milestone 8")


def is_triton_available() -> bool:
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False