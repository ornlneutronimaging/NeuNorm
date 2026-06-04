"""
Numba compatibility module with graceful fallback.

Provides Numba-like decorators that work whether or not Numba is installed.
When Numba is available, functions are JIT-compiled for performance.
When Numba is not available, the decorators are no-ops and functions
run as regular Python (slower, but functional).

Usage
-----
>>> from neunorm.utils._numba_compat import jit, njit, HAS_NUMBA
>>>
>>> @jit(nopython=True)
... def fast_loop(arr):
...     total = 0.0
...     for x in arr:
...         total += x
...     return total
>>>
>>> # Check if Numba is being used
>>> if HAS_NUMBA:
...     print("Using Numba JIT compilation")
... else:
...     print("Running in pure Python mode")

Notes
-----
- Functions decorated with @jit or @njit work identically with or without Numba
- Numba provides 50-100x speedup for loop-heavy code
- Install Numba with: pip install neunorm[performance]
"""

import functools
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# Try to import Numba
try:
    from numba import jit as _numba_jit
    from numba import njit as _numba_njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    _numba_jit = None
    _numba_njit = None


def _make_wrapper(func: F) -> F:
    """Create a wrapper function that preserves the original function's behavior."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


class _NoOpJit:
    """
    No-op JIT decorator class for when Numba is not available.

    Handles all decorator patterns:
    - @jit (direct decoration)
    - @jit() (called without arguments)
    - @jit(nopython=True) (called with arguments)
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Handle both direct decoration and called decorator patterns."""
        # Case 1: @jit or @jit(func) - direct decoration without parentheses
        # args[0] is the function being decorated
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return _make_wrapper(args[0])

        # Case 2: @jit() or @jit(nopython=True, cache=True, etc.)
        # Return self as a decorator (will be called again with the function)
        return self

    def __repr__(self) -> str:
        return "<no-op jit decorator (numba not available)>"


_noop_jit = _NoOpJit()


if HAS_NUMBA:
    # Use real Numba decorators
    jit = _numba_jit
    njit = _numba_njit
else:
    # Use no-op fallback decorators (same instance handles both)
    jit = _noop_jit
    njit = _noop_jit


__all__ = ["HAS_NUMBA", "jit", "njit"]
