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
- Save this prompt to: `auto_reports/prompts/prompt_P3_jit_remaining_functions.md`
- Save final report to: `auto_reports/report_P3_jit_remaining_functions.md`
- Link prompt at top of report

---
## Task: P3 — JIT-Compile Remaining Hot Functions

### Objective
Apply Numba JIT compilation to the remaining hot functions in `pulse_reconstruction.py` that were identified in the optimization research.

### Context
- P1 completed: Numba compatibility module in place
- P2 completed: `_refine_rollover_boundaries()` now uses JIT-compiled helper
- Remaining functions to optimize: `_detect_rollovers`, `_clean_clustered_rollovers`, and any other functions with significant loop overhead

### WHAT to Implement

1. **Audit `pulse_reconstruction.py` for remaining optimization targets**
   - Identify functions with loops or repeated array operations
   - Prioritize by call frequency and computational cost

2. **Apply JIT compilation where beneficial**
   - Use the compatibility decorators from P1
   - May need to extract inner loops to helper functions (as done in P2)
   - Ensure Numba compatibility (no Python objects, supported NumPy operations)

3. **Maintain backward compatibility**
   - All function signatures must remain unchanged
   - Results must be identical with or without Numba
   - All existing tests must continue to pass

4. **Write tests for JIT/non-JIT equivalence**
   - Verify correctness with realistic data
   - Test edge cases

### WHY
- Complete the JIT optimization coverage for pulse reconstruction
- Each optimized function contributes to overall speedup
- Prepares for P4 (parallel multi-chip processing)

### Acceptance Criteria
- [ ] All hot functions in `pulse_reconstruction.py` audited
- [ ] JIT applied to functions where beneficial
- [ ] Falls back to pure Python when Numba unavailable
- [ ] Results identical with and without Numba
- [ ] All existing tests pass
- [ ] New tests verify JIT/non-JIT equivalence for each optimized function

---
## Report Requirements

### Save Locations
- Prompt: `auto_reports/prompts/prompt_P3_jit_remaining_functions.md`
- Report: `auto_reports/report_P3_jit_remaining_functions.md`

### Report Contents
- Link to prompt file
- List of functions audited and decision (JIT/no JIT/why)
- Files modified
- Test count and results
- Commits made
- Any Numba-specific challenges encountered
- Ready for P4 confirmation
