export const meta = {
  name: 'bughunt-deep',
  description:
    'Multi-level, HIGH-REDUNDANCY adversarial bug hunt over src/neunorm. A system-MAP pass builds shared context (data flow, scipp/dim/unit/variance/mask conventions, detector_shape, shared constants, callers). That context feeds, per functional PACKAGE, MANY independent whole-package reviewers — several Claude passes across logic/contracts/holistic lenses PLUS several cross-family Codex passes (LLMs are stochastic; different agents find different bugs, more so across model families) — plus cross-PACKAGE integration reviewers on the seams (pipeline ordering, data-model contracts, uncertainty/mask flow), each also multi-pass. Then cross-finder dedup and a 3-skeptic verification panel with reachability grading. Deliberately token-heavy: one exhaustive round beats repeated shallow ones. For pre-release audits.',
  phases: [
    { title: 'Map', detail: 'system map: packages, files, callers, shared contracts' },
    { title: 'Find', detail: 'per-package finders (with caller context) + cross-package integration finders' },
    { title: 'Verify', detail: '3 independent skeptics per finding; majority keeps it, reachability graded' },
    { title: 'Synthesize', detail: 'group confirmed findings by severity + reachability' },
  ],
}

// Optional scope note from the caller. Defaults to a full audit of src/neunorm.
const scope = typeof args === 'string' && args.trim() ? args.trim() : 'all of src/neunorm'

const rank = { P0: 0, P1: 1, P2: 2, nit: 3 }
const reachOrder = { common: 0, 'edge-case': 1, latent: 2, unreachable: 3 }
// Redundancy is the POINT, not waste: LLMs are stochastic, so several independent
// passes over the SAME whole package find different bugs, and cross-family (Codex)
// passes find more still. One exhaustive round beats repeated shallow ones.
const CLAUDE_PASSES = 2 // independent Claude passes PER lens, per package (whole package each)
const CODEX_PASSES = 2 // independent cross-family Codex passes per package (whole package each)
const INTEGRATION_PASSES = 2 // independent Claude passes per cross-package integration lane

// Model tiering. Discovery is the hard research (top model); the Map is mechanical
// enumeration + a summary (cheap model is fine, but it is load-bearing context for every
// downstream agent, so use Sonnet, not Haiku); verification is a bounded refute-and-grade
// task where robustness comes from the 3-skeptic majority, so a cheaper model is fine and
// it is ~89% of the agents — the real cost lever. Codex lanes call the GPT CLI, so the
// Claude wrapper that shells out and reshapes the output can be cheap too.
const MODEL_MAP = 'sonnet' // system map / orientation
const MODEL_FIND = 'opus' // per-package + integration bug discovery (complex research)
const MODEL_VERIFY = 'sonnet' // skeptic panel (majority of 3 gives robustness)
const MODEL_CODEX = 'sonnet' // Claude wrapper around the codex CLI

// ---------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------
const MAP_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    overview: {
      type: 'string',
      description:
        'shared context every auditor must read to avoid tunnel vision: end-to-end data flow (loaders -> processing/tof -> pipelines -> exporters), scipp conventions (dims, units, variance & mask propagation), the detector_shape convention, key shared constants (e.g. TOF clock periods in utils/constants.py), and any cross-cutting contracts',
    },
    codexAvailable: { type: 'boolean', description: "true if 'command -v codex' succeeds" },
    packages: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          name: { type: 'string' },
          dir: { type: 'string', description: 'package dir, e.g. src/neunorm/tof' },
          purpose: { type: 'string' },
          files: {
            type: 'array',
            items: { type: 'string' },
            description: 'the substantive .py files in this package (exclude empty __init__.py and _version.py)',
          },
          usedBy: {
            type: 'array',
            items: { type: 'string' },
            description: 'other packages/modules that import or call this one (its callers)',
          },
        },
        required: ['name', 'dir', 'purpose', 'files', 'usedBy'],
      },
    },
  },
  required: ['overview', 'codexAvailable', 'packages'],
}

const FINDINGS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          severity: { type: 'string', enum: ['P0', 'P1', 'P2', 'nit'] },
          file: { type: 'string', description: 'path, ideally under src/neunorm' },
          line: { type: 'integer', description: 'line number, or 0 if unknown' },
          title: { type: 'string' },
          why: { type: 'string', description: 'why this is a bug, grounded in the actual code AND how it is used' },
          trigger: { type: 'string', description: 'the exact input/condition that triggers it, and whether default/real usage reaches it' },
          fix: { type: 'string', description: 'concrete fix suggestion' },
        },
        required: ['severity', 'file', 'line', 'title', 'why', 'trigger', 'fix'],
      },
    },
  },
  required: ['findings'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    real: { type: 'boolean', description: 'true only if the code clearly confirms a genuine bug' },
    reachability: {
      type: 'string',
      enum: ['common', 'edge-case', 'latent', 'unreachable'],
      description:
        'how reachable on valid/real inputs: common = default/real usage hits it; edge-case = only unusual-but-valid inputs; latent = only non-default params/configs; unreachable = cannot happen given how the code is actually used (e.g. a non-square-detector path when every real detector is square)',
    },
    confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
    reason: { type: 'string' },
  },
  required: ['real', 'reachability', 'confidence', 'reason'],
}

// ---------------------------------------------------------------------------
// Level 0 — Map: build the shared system context so finders see the big picture.
// ---------------------------------------------------------------------------
phase('Map')
log(`Multi-level bug hunt scope: ${scope}`)
const map = await agent(
  `Build a system map of src/neunorm for scope: ${scope}. Use find/grep to inspect the tree. ` +
    `Identify the functional PACKAGES (subdirectories of src/neunorm). For each package return: name, dir (e.g. src/neunorm/tof), one-line purpose, its substantive .py files (exclude empty __init__.py and _version.py), and usedBy (which other packages import or call it). ` +
    `Also write a thorough "overview" an auditor must read to avoid tunnel vision: the end-to-end data flow (loaders -> processing/tof -> pipelines -> exporters), the scipp conventions this codebase relies on (dims, units, variance and mask propagation), the detector_shape convention ((x,y) vs (y,x), square vs non-square in practice), key shared constants such as the TOF clock periods in utils/constants.py, and any cross-cutting contracts (e.g. "normalize_transmission documents that zero-OB must be masked by the caller"). ` +
    `Finally report codexAvailable: does 'command -v codex' succeed?`,
  { label: 'map', phase: 'Map', schema: MAP_SCHEMA, agentType: 'bug-hunter', model: MODEL_MAP },
)

const overview = (map?.overview || '').trim()
const packages = (map?.packages || []).filter((p) => p && p.dir && (p.files || []).length)
const codexAvailable = !!map?.codexAvailable
log(
  `${packages.length} packages mapped (${packages.reduce((n, p) => n + p.files.length, 0)} files); ` +
    (codexAvailable ? 'Codex available' : 'Codex not installed (Claude-only)'),
)

// ---------------------------------------------------------------------------
// Level 1 — Find: per-package finders (with caller context) + integration finders.
// ---------------------------------------------------------------------------
phase('Find')

const LENSES = [
  {
    key: 'logic',
    focus:
      'algorithmic correctness; numerical issues (dtype/precision, float32 catastrophic cancellation, off-by-one in bins, division by zero, inf/NaN, truncation); and edge cases (empty / single-element / degenerate / zero-size inputs).',
  },
  {
    key: 'contracts',
    focus:
      'API/docstring contract mismatches; scipp shape/dim/unit/dtype handling; variance and mask propagation; error handling; file/HDF5 handle lifetime; and assumptions about NeXus / metadata keys and array layout (row- vs column-major, (x,y) vs (y,x)).',
  },
  {
    key: 'holistic',
    focus:
      'anything that looks wrong, surprising, fragile, or inconsistent with the documented physics/behavior — do NOT restrict yourself to a category. Trust your judgment about how this package could misbehave on real data, and look especially for whatever the logic and contracts lenses might miss.',
  },
]

const context = (p) =>
  `SYSTEM OVERVIEW (read for context; do not re-derive):\n${overview}\n\n` +
  `You are auditing the "${p.name}" package — ${p.dir} (${p.purpose}).\n` +
  `Files in this package: ${p.files.join(', ')}.\n` +
  `This package is used by: ${(p.usedBy || []).join(', ') || 'the pipelines / top level'}.`

const finderThunks = []
for (const p of packages) {
  const callers = (p.usedBy || []).join(', ') || 'the pipelines / top level'
  // Several independent Claude passes per lens, each auditing the WHOLE package
  // (no file chunking — full context, redundant coverage across stochastic runs).
  for (const lens of LENSES) {
    for (let k = 0; k < CLAUDE_PASSES; k++) {
      finderThunks.push(() =>
        agent(
          `${context(p)}\n\n` +
            `Audit the WHOLE ${p.name} package through the "${lens.key}" lens: ${lens.focus}\n` +
            `READ every file in the package, its tests, AND its callers (${callers}) so you understand how these functions are ACTUALLY used — never flag a bug without understanding the bigger picture. ` +
            `(Independent review pass #${k + 1}: other agents audit this same package in parallel; LLMs are stochastic, so dig for what a different reviewer would overlook, not just the obvious.) ` +
            `Ground every finding in specific lines. Fill "trigger" with the exact condition that hits it and whether default/real usage reaches it — a defect only reachable on contrived or impossible inputs is at most P2/nit, not P1.`,
          { label: `find:${p.name}:${lens.key}#${k + 1}`, phase: 'Find', schema: FINDINGS_SCHEMA, agentType: 'bug-hunter', model: MODEL_FIND },
        ),
      )
    }
  }
  // Several independent cross-family Codex passes per package.
  if (codexAvailable) {
    for (let k = 0; k < CODEX_PASSES; k++) {
      finderThunks.push(() =>
        agent(
          `Provide an INDEPENDENT bug review of the ${p.name} package (${p.dir}) using the Codex CLI (independent pass #${k + 1}). ` +
            `Run it READ-ONLY (no file edits): codex exec "Review the files in ${p.dir} (${p.files.join(', ')}) of this scipp-based neutron imaging library for correctness, numerical, and edge-case bugs, considering how they are used by ${callers}. For each finding give severity (P0/P1/P2/nit), file:line, why, the trigger condition, and a fix. Do not modify any files." ` +
            `Then translate Codex's output into the schema (keep only findings about files in ${p.dir}, set each "trigger"). If codex is unavailable, return an empty findings list. Never edit files.`,
          { label: `find:${p.name}:codex#${k + 1}`, phase: 'Find', schema: FINDINGS_SCHEMA, model: MODEL_CODEX },
        ),
      )
    }
  }
}

// Cross-package integration finders — the seams where per-package audits go blind.
const INTEGRATIONS = [
  {
    key: 'pipelines-e2e',
    prompt:
      'Audit the END-TO-END MARS/VENUS pipelines (src/neunorm/pipelines). Check the ORDER of corrections (dark / gamma / ROI / dead-pixel / normalize) and whether shapes, dims, units, variances and masks stay consistent as data flows loaders -> processing/tof -> exporters. Flag steps applied in the wrong order, to the wrong operand (e.g. a mask or correction applied to sample but not to the OB denominator), or that silently drop variance/mask.',
  },
  {
    key: 'data-contracts',
    prompt:
      'Audit the data-model contracts (EventData, BinningConfig in src/neunorm/data_models) where they are PRODUCED (loaders/tof) versus CONSUMED (tof/pipelines): dtype/shape/unit mismatches, fields that are set but ignored, or assumptions that disagree between producer and consumer (e.g. event_id layout, tof units ns vs us).',
  },
  {
    key: 'uncertainty-mask-flow',
    prompt:
      'Audit variance/uncertainty and mask propagation across the WHOLE chain (uncertainty_calculator, reference_preparer, dark_corrector, normalizer, exporters). Find places where variances or masks are silently dropped, double-counted, broadcast away, or computed inconsistently — automatic uncertainty propagation is the library headline.',
  },
]
for (const it of INTEGRATIONS) {
  for (let k = 0; k < INTEGRATION_PASSES; k++) {
    finderThunks.push(() =>
      agent(
        `SYSTEM OVERVIEW (read for context):\n${overview}\n\n` +
          `CROSS-PACKAGE integration audit (independent pass #${k + 1}). ${it.prompt}\n` +
          `Read across the relevant packages and their tests. Ground every finding in specific lines; fill "trigger" with the condition and whether default/real usage reaches it.`,
        { label: `find:integration:${it.key}#${k + 1}`, phase: 'Find', schema: FINDINGS_SCHEMA, agentType: 'bug-hunter', model: MODEL_FIND },
      ),
    )
  }
  if (codexAvailable) {
    finderThunks.push(() =>
      agent(
        `Independent Codex review for a CROSS-PACKAGE integration concern. Run READ-ONLY: codex exec "In this scipp-based neutron imaging library (src/neunorm), ${it.prompt} Give severity, file:line, why, trigger, and fix for each finding. Do not modify files." ` +
          `Translate into the schema, set each "trigger". If codex is unavailable, return an empty findings list. Never edit files.`,
        { label: `find:integration:${it.key}:codex`, phase: 'Find', schema: FINDINGS_SCHEMA, model: MODEL_CODEX },
      ),
    )
  }
}

log(
  `${finderThunks.length} finder agents over ${packages.length} packages + ${INTEGRATIONS.length} integration lanes ` +
    `(${CLAUDE_PASSES} Claude passes x ${LENSES.length} lenses${codexAvailable ? ` + ${CODEX_PASSES} Codex passes` : ''} per package)`,
)
const finderResults = (await parallel(finderThunks)).filter(Boolean)

// Dedup across ALL finders (barrier justified: verify each unique issue once).
const norm = (s) => (s || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim()
const rawCount = finderResults.reduce((n, r) => n + (r.findings?.length || 0), 0)
const clusters = new Map()
for (const r of finderResults) {
  for (const f of r.findings || []) {
    if (!f || !f.file) continue
    const key = f.line && f.line > 0 ? `${f.file}:${f.line}` : `${f.file}:${norm(f.title).slice(0, 50)}`
    const prev = clusters.get(key)
    if (!prev) {
      clusters.set(key, { ...f, foundBy: 1 })
    } else {
      prev.foundBy += 1
      if ((rank[f.severity] ?? 9) < (rank[prev.severity] ?? 9)) prev.severity = f.severity
      if ((f.why || '').length > (prev.why || '').length) {
        prev.why = f.why
        prev.title = f.title
      }
      if ((f.trigger || '').length > (prev.trigger || '').length) prev.trigger = f.trigger
      if ((f.fix || '').length > (prev.fix || '').length) prev.fix = f.fix
    }
  }
}
const candidates = [...clusters.values()]
log(`${rawCount} raw findings -> ${candidates.length} unique candidates`)

// ---------------------------------------------------------------------------
// Verify — 3 skeptics per candidate, each given the system overview so they can
// judge reachability correctly (not just "does it crash in isolation").
// ---------------------------------------------------------------------------
phase('Verify')
const judged = await parallel(
  candidates.map((f) => () =>
    parallel(
      ['correctness', 'reproduce', 'edge-cases'].map((lens) => () =>
        agent(
          `You are a skeptical reviewer. Using the "${lens}" lens, try to REFUTE this finding by reading the ACTUAL code plus its tests and CALLERS in the repo. ` +
            `Use this system context to judge how the code is really used:\n${overview}\n\n` +
            `Decide (a) real: is it a genuine defect; and (b) reachability on valid/real inputs — "common", "edge-case", "latent", or "unreachable" (cannot happen given how the code is actually used). ` +
            `Default real=false unless the code clearly confirms a genuine defect. Finding: ${JSON.stringify(f)}`,
          { label: `verify:${(f.file || '').split('/').pop()}:${f.line || '?'}`, phase: 'Verify', schema: VERDICT_SCHEMA, model: MODEL_VERIFY },
        ),
      ),
    ).then((verdicts) => {
      const votes = verdicts.filter(Boolean)
      const real = votes.filter((v) => v.real).length >= 2 // majority of 3 skeptics
      const reaches = votes.filter((v) => v.real).map((v) => v.reachability).filter(Boolean)
      const reachability = reaches.sort((a, b) => (reachOrder[a] ?? 9) - (reachOrder[b] ?? 9))[0] || 'unknown'
      return { ...f, real, reachability, votes }
    }),
  ),
)

// ---------------------------------------------------------------------------
// Synthesize — confirmed findings, ranked by severity then reachability.
// ---------------------------------------------------------------------------
phase('Synthesize')
const confirmed = judged
  .filter((j) => j.real)
  .sort(
    (a, b) =>
      (rank[a.severity] ?? 9) - (rank[b.severity] ?? 9) ||
      (reachOrder[a.reachability] ?? 9) - (reachOrder[b.reachability] ?? 9) ||
      (a.file || '').localeCompare(b.file || ''),
  )
log(`Confirmed ${confirmed.length} / ${candidates.length} candidate findings`)

return {
  scope,
  packagesAudited: packages.length,
  finderAgents: finderThunks.length,
  confirmed,
  counts: {
    rawFindings: rawCount,
    candidates: candidates.length,
    confirmed: confirmed.length,
    dropped: candidates.length - confirmed.length,
    bySeverity: confirmed.reduce((a, f) => ((a[f.severity] = (a[f.severity] || 0) + 1), a), {}),
    byReachability: confirmed.reduce((a, f) => ((a[f.reachability] = (a[f.reachability] || 0) + 1), a), {}),
  },
}
