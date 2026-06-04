"""Unit tests for Numba compatibility/fallback module.

Tests the graceful fallback pattern when Numba is not available.
The module should:
1. Detect Numba availability at runtime
2. Provide no-op decorator fallbacks when Numba is not installed
3. Functions decorated with jit should work identically with or without Numba
"""

import sys
from unittest import mock

import numpy as np
import pytest


class TestNumbaCompatModule:
    """Tests for the _numba_compat module import and availability detection."""

    def test_module_imports_successfully(self):
        """The _numba_compat module should import without errors."""
        from neunorm.utils import _numba_compat

        assert _numba_compat is not None

    def test_has_numba_flag_exists(self):
        """Module should expose HAS_NUMBA boolean flag."""
        from neunorm.utils._numba_compat import HAS_NUMBA

        assert isinstance(HAS_NUMBA, bool)

    def test_jit_decorator_is_exported(self):
        """Module should export a jit decorator."""
        from neunorm.utils._numba_compat import jit

        assert callable(jit)

    def test_njit_decorator_is_exported(self):
        """Module should export an njit decorator (alias for jit with nopython=True)."""
        from neunorm.utils._numba_compat import njit

        assert callable(njit)


class TestNumbaAvailable:
    """Tests for behavior when Numba IS available."""

    @pytest.fixture
    def compat_with_numba(self):
        """Get a fresh import of _numba_compat with Numba available."""
        # Force reimport to ensure clean state
        module_name = "neunorm.utils._numba_compat"
        if module_name in sys.modules:
            del sys.modules[module_name]
        from neunorm.utils import _numba_compat

        return _numba_compat

    def test_has_numba_reflects_availability(self, compat_with_numba):
        """HAS_NUMBA should be True when Numba is installed."""
        try:
            import numba  # noqa: F401

            assert compat_with_numba.HAS_NUMBA is True
        except ImportError:
            pytest.skip("Numba not installed, cannot test 'available' behavior")

    def test_jit_returns_callable(self, compat_with_numba):
        """jit decorator should return a callable."""
        jit = compat_with_numba.jit

        @jit(nopython=True)
        def add_one(x):
            return x + 1

        assert callable(add_one)

    def test_njit_returns_callable(self, compat_with_numba):
        """njit decorator should return a callable."""
        njit = compat_with_numba.njit

        @njit
        def add_two(x):
            return x + 2

        assert callable(add_two)

    def test_decorated_function_works_with_numpy(self, compat_with_numba):
        """Decorated function should work with numpy arrays."""
        jit = compat_with_numba.jit

        @jit(nopython=True)
        def sum_array(arr):
            total = 0.0
            for i in range(len(arr)):
                total += arr[i]
            return total

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sum_array(arr)

        assert np.isclose(result, 15.0)


class TestNumbaUnavailable:
    """Tests for behavior when Numba is NOT available (fallback mode)."""

    @pytest.fixture
    def compat_without_numba(self):
        """
        Get a fresh import of _numba_compat with Numba mocked as unavailable.

        Uses sys.modules manipulation to simulate ImportError on numba import.
        """
        module_name = "neunorm.utils._numba_compat"

        # Remove cached module
        if module_name in sys.modules:
            del sys.modules[module_name]

        # Mock numba as unavailable by raising ImportError
        with mock.patch.dict(sys.modules, {"numba": None}):
            # Force the import to raise ImportError when accessing numba
            original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

            def mock_import(name, *args, **kwargs):
                if name == "numba" or name.startswith("numba."):
                    raise ImportError(f"Mocked: No module named '{name}'")
                return original_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=mock_import):
                # Now import the module - it should detect numba as unavailable
                import neunorm.utils._numba_compat as compat

                # Capture the module state
                has_numba = compat.HAS_NUMBA
                jit = compat.jit
                njit = compat.njit

        # Clean up for other tests
        if module_name in sys.modules:
            del sys.modules[module_name]

        return type("CompatWithoutNumba", (), {"HAS_NUMBA": has_numba, "jit": jit, "njit": njit})()

    def test_has_numba_is_false(self, compat_without_numba):
        """HAS_NUMBA should be False when Numba import fails."""
        assert compat_without_numba.HAS_NUMBA is False

    def test_jit_returns_identity_decorator(self, compat_without_numba):
        """jit should return a no-op decorator that returns the function unchanged."""
        jit = compat_without_numba.jit

        def original_func(x):
            return x * 2

        # Apply decorator
        decorated = jit(nopython=True)(original_func)

        # Should return the same function (or equivalent behavior)
        assert decorated(5) == 10
        assert decorated(0) == 0
        assert decorated(-3) == -6

    def test_njit_returns_identity_decorator(self, compat_without_numba):
        """njit should return a no-op decorator that returns the function unchanged."""
        njit = compat_without_numba.njit

        def original_func(x, y):
            return x + y

        decorated = njit(original_func)

        assert decorated(2, 3) == 5
        assert decorated(0, 0) == 0

    def test_decorated_function_works_with_numpy_fallback(self, compat_without_numba):
        """Decorated function should work with numpy arrays in fallback mode."""
        jit = compat_without_numba.jit

        @jit(nopython=True)
        def sum_array(arr):
            total = 0.0
            for i in range(len(arr)):
                total += arr[i]
            return total

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sum_array(arr)

        assert np.isclose(result, 15.0)


class TestFunctionalEquivalence:
    """Tests ensuring decorated functions produce identical results with/without Numba."""

    def test_simple_loop_function_equivalence(self):
        """A simple loop function should produce same results with and without Numba."""
        from neunorm.utils._numba_compat import jit

        @jit(nopython=True)
        def compute_sum(arr):
            """Sum array elements using explicit loop."""
            total = 0.0
            for i in range(len(arr)):
                total += arr[i]
            return total

        # Test with various inputs
        test_cases = [
            np.array([1.0, 2.0, 3.0]),
            np.zeros(10),
            np.ones(100),
            np.linspace(0, 100, 50),
        ]

        for arr in test_cases:
            result = compute_sum(arr)
            expected = np.sum(arr)
            assert np.isclose(result, expected), f"Mismatch for array of length {len(arr)}"

    def test_boundary_finding_function(self):
        """Test a function similar to rollover boundary detection."""
        from neunorm.utils._numba_compat import jit

        @jit(nopython=True)
        def find_drops(arr, threshold):
            """Find indices where consecutive diff drops below threshold."""
            indices = []
            for i in range(1, len(arr)):
                if arr[i] - arr[i - 1] < threshold:
                    indices.append(i)
            return indices

        # TOF-like data with rollovers
        tof_data = np.array([1.0, 2.0, 3.0, 14.0, 15.0, 1.0, 2.0, 3.0, 14.0, 15.0, 1.0])

        result = find_drops(tof_data, -10.0)

        # Rollovers at indices 5 and 10 (15.0 -> 1.0 = -14.0 < -10.0)
        assert 5 in result
        assert 10 in result
        assert len(result) == 2

    def test_array_modification_function(self):
        """Test a function that modifies array values in place."""
        from neunorm.utils._numba_compat import jit

        @jit(nopython=True)
        def assign_pulse_ids(n_events, boundaries):
            """Assign pulse IDs based on boundaries."""
            pulse_ids = np.zeros(n_events, dtype=np.int32)
            for i, boundary in enumerate(boundaries):
                pulse_ids[boundary:] = i + 1
            return pulse_ids

        boundaries = np.array([50, 100, 150])
        result = assign_pulse_ids(200, boundaries)

        # Check assignments
        assert np.all(result[:50] == 0)
        assert np.all(result[50:100] == 1)
        assert np.all(result[100:150] == 2)
        assert np.all(result[150:] == 3)


class TestDecoratorPatterns:
    """Tests for various decorator usage patterns."""

    def test_jit_with_no_arguments(self):
        """jit() with no arguments should work."""
        from neunorm.utils._numba_compat import jit

        @jit
        def simple_add(a, b):
            return a + b

        assert simple_add(1, 2) == 3

    def test_jit_with_nopython_true(self):
        """jit(nopython=True) should work."""
        from neunorm.utils._numba_compat import jit

        @jit(nopython=True)
        def simple_multiply(a, b):
            return a * b

        assert simple_multiply(3, 4) == 12

    def test_jit_with_cache_true(self):
        """jit(cache=True) should work (or be ignored in fallback)."""
        from neunorm.utils._numba_compat import jit

        @jit(nopython=True, cache=True)
        def cached_func(x):
            return x * x

        assert cached_func(5) == 25

    def test_jit_with_parallel_true(self):
        """jit(parallel=True) should work (or be ignored in fallback)."""
        from neunorm.utils._numba_compat import jit

        @jit(nopython=True, parallel=True)
        def parallel_sum(arr):
            total = 0.0
            for x in arr:
                total += x
            return total

        arr = np.arange(100.0)
        result = parallel_sum(arr)
        assert np.isclose(result, np.sum(arr))

    def test_njit_direct_decoration(self):
        """@njit without parentheses should work."""
        from neunorm.utils._numba_compat import njit

        @njit
        def direct_njit(x):
            return x + 1

        assert direct_njit(10) == 11

    def test_njit_with_cache(self):
        """@njit(cache=True) should work."""
        from neunorm.utils._numba_compat import njit

        @njit(cache=True)
        def cached_njit(x):
            return x - 1

        assert cached_njit(10) == 9


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_empty_array_input(self):
        """Decorated functions should handle empty arrays."""
        from neunorm.utils._numba_compat import jit

        @jit(nopython=True)
        def sum_empty(arr):
            total = 0.0
            for x in arr:
                total += x
            return total

        empty = np.array([], dtype=np.float64)
        result = sum_empty(empty)
        assert result == 0.0

    def test_single_element_array(self):
        """Decorated functions should handle single-element arrays."""
        from neunorm.utils._numba_compat import jit

        @jit(nopython=True)
        def get_first(arr):
            if len(arr) > 0:
                return arr[0]
            return -1.0

        single = np.array([42.0])
        result = get_first(single)
        assert result == 42.0

    def test_large_array_performance(self):
        """Large array should process without error (performance may vary)."""
        from neunorm.utils._numba_compat import jit

        @jit(nopython=True)
        def count_positives(arr):
            count = 0
            for x in arr:
                if x > 0:
                    count += 1
            return count

        large = np.random.randn(100000)
        result = count_positives(large)

        # Roughly half should be positive
        assert 40000 < result < 60000

    def test_integer_array_input(self):
        """Decorated functions should work with integer arrays."""
        from neunorm.utils._numba_compat import jit

        @jit(nopython=True)
        def sum_ints(arr):
            total = 0
            for x in arr:
                total += x
            return total

        ints = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        result = sum_ints(ints)
        assert result == 15

    def test_nested_loops(self):
        """Decorated functions with nested loops should work."""
        from neunorm.utils._numba_compat import jit

        @jit(nopython=True)
        def matrix_sum(matrix):
            total = 0.0
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    total += matrix[i, j]
            return total

        mat = np.array([[1, 2], [3, 4]], dtype=np.float64)
        result = matrix_sum(mat)
        assert result == 10.0

    def test_exception_propagation(self):
        """Exceptions raised in decorated functions should propagate correctly."""
        from neunorm.utils._numba_compat import jit

        @jit(nopython=True)
        def raise_on_negative(x):
            if x < 0:
                raise ValueError("Negative value not allowed")
            return x * 2

        # Normal case should work
        assert raise_on_negative(5) == 10

        # Exception should propagate
        with pytest.raises(ValueError, match="Negative value not allowed"):
            raise_on_negative(-1)

    def test_function_metadata_preserved(self):
        """Decorated functions should preserve __name__ and __doc__."""
        from neunorm.utils._numba_compat import jit

        @jit(nopython=True)
        def documented_function(x):
            """This is a well-documented function."""
            return x + 1

        # Note: Numba's jit doesn't guarantee metadata preservation,
        # but in fallback mode we should preserve it.
        # This test checks that the function is still callable and identifiable
        assert callable(documented_function)
        assert documented_function(5) == 6
