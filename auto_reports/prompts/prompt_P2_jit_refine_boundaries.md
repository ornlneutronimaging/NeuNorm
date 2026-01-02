---
## Behavior Control

### Repository Setup
- Repository: NeuNorm (`/SNS/users/8cz/github.com/NeuNorm`)
- Continue on branch: `feature/numba-optimization`
- Read `Claude.MD` before starting any work

### Development Protocol
- Follow TDD: Write tests first, then implement
- Request independent agent review for tests before implementing
- Request independent agent review for implementation before committing
- Commit with conventional prefixes: `feat:`, `test:`, `refactor:`

### Archiving
- Save this prompt to: `auto_reports/prompts/prompt_P2_jit_refine_boundaries.md`
- Save final report to: `auto_reports/report_P2_jit_refine_boundaries.md`
- Link prompt at top of report

---
## Task: P2 — JIT-Compile `_refine_rollover_boundaries()`

### Objective
Apply Numba JIT compilation to `_refine_rollover_boundaries()` in `pulse_reconstruction.py` — the identified bottleneck function.

### Context
- P1 completed: Numba compatibility module exists at `neunorm.utils._numba_compat`
- The `_refine_rollover_boundaries()` function contains nested loops that are the primary bottleneck
- Current processing: ~11 hours for 7.2 billion events
- Target: 50-100x speedup on this function

### WHAT to Implement

1. **Refactor `_refine_rollover_boundaries()` for Numba compatibility**
   - Use the compatibility decorators from P1
   - Ensure the function works with and without Numba
   - May need to extract inner computation to a separate JIT-compiled helper function
   - Numba has restrictions (no Python objects, limited NumPy API) — handle appropriately

2. **Maintain backward compatibility**
   - Function signature must remain unchanged
   - Results must be identical with or without Numba
   - Existing tests must continue to pass

3. **Write performance and correctness tests**
   - Test that results match between JIT and non-JIT versions
   - Test with realistic data sizes
   - Verify edge cases still work

### WHY
- This function is the primary bottleneck (identified in optimization research)
- Nested Python loops have high overhead that Numba eliminates
- Expected 50-100x speedup on this function alone

### Acceptance Criteria
- [ ] `_refine_rollover_boundaries()` uses Numba JIT when available
- [ ] Falls back to pure Python when Numba unavailable
- [ ] Results are identical with and without Numba
- [ ] All existing pulse reconstruction tests pass
- [ ] New tests verify JIT/non-JIT equivalence
- [ ] No regression in functionality

---
## Report Requirements

### Save Locations
- Prompt: `auto_reports/prompts/prompt_P2_jit_refine_boundaries.md`
- Report: `auto_reports/report_P2_jit_refine_boundaries.md`

### Report Contents
- Link to prompt file
- Files modified
- Test count and results
- Commits made
- Any Numba-specific challenges encountered
- Performance improvement estimate (if measurable with small test data)
- Ready for P3 confirmation
