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
- Save this prompt to: `auto_reports/prompts/prompt_P4_parallel_processing.md`
- Save final report to: `auto_reports/report_P4_parallel_processing.md`
- Link prompt at top of report

---
## Task: P4 — Add Parallel Multi-Chip Processing

### Objective
Add parallel processing for multi-chip pulse reconstruction, so that all 4 chips can be processed simultaneously instead of sequentially.

### Context
- P1-P3 completed: JIT compilation in place for hot functions
- Current behavior: `reconstruct_pulse_ids()` processes chips sequentially in a for-loop
- Hardware: 32-core CPU available, 4 chips to process
- Expected speedup: ~3.5-4x from parallelization alone

### WHAT to Implement

1. **Add parallel processing option to `reconstruct_pulse_ids()`**
   - Use `joblib` (already available via scikit-learn) or `multiprocessing`
   - Process each chip independently in parallel
   - Make parallelization optional (parameter to enable/disable)
   - Default behavior should be safe (consider whether parallel should be default or opt-in)

2. **Handle data correctly**
   - Each chip's data is independent — no synchronization needed
   - Results must be combined correctly into final `pulse_ids` array
   - Memory considerations: each chip processes ~1.7B events

3. **Maintain backward compatibility**
   - Function signature can add optional parameter, but defaults must preserve current behavior
   - Results must be identical whether parallel or sequential
   - All existing tests must pass

4. **Write tests**
   - Test parallel vs sequential equivalence
   - Test with different numbers of chips
   - Test edge cases (single chip, empty data)

### WHY
- Currently 4 chips process in ~11 hours sequentially
- With P1-P3 JIT optimizations: expect ~15-30 minutes sequential
- With P4 parallel: expect ~4-8 minutes total
- Parallelization is low-risk since chips are independent

### Acceptance Criteria
- [ ] `reconstruct_pulse_ids()` supports parallel multi-chip processing
- [ ] Parallelization is controllable (on/off)
- [ ] Results identical between parallel and sequential
- [ ] All existing tests pass
- [ ] New tests verify parallel/sequential equivalence
- [ ] Works correctly with both Numba available and unavailable

---
## Report Requirements

### Save Locations
- Prompt: `auto_reports/prompts/prompt_P4_parallel_processing.md`
- Report: `auto_reports/report_P4_parallel_processing.md`

### Report Contents
- Link to prompt file
- Implementation approach chosen (joblib vs multiprocessing)
- Files modified
- Test count and results
- Commits made
- Any challenges with parallel processing
- Ready for P5 confirmation
