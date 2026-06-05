export const meta = {
  name: 'bughunt-deep',
  description:
    'Exhaustive, adversarially-verified bug hunt over src/neunorm: parallel per-subsystem finders plus an optional Codex second opinion, then multi-skeptic verification to drop false positives. Token-heavy; for pre-release audits.',
  phases: [
    { title: 'Find', detail: 'parallel finders per subsystem + a Codex lane' },
    { title: 'Verify', detail: '3 independent skeptics per finding; majority keeps it' },
    { title: 'Synthesize', detail: 'group confirmed findings by severity' },
  ],
}

// Optional scope note from the caller (e.g. "focus on the VENUS TPX3 pipeline" or
// a diff range). Defaults to a full audit of src/neunorm.
const scope = typeof args === 'string' && args.trim() ? args.trim() : 'all of src/neunorm'

const SUBSYSTEMS = [
  {
    key: 'loaders',
    focus:
      'TIFF/FITS/event(HDF5)/metadata(NeXus) loaders: shape & dtype handling, NeXus key assumptions, empty/single-frame stacks, file-handle lifetime',
  },
  {
    key: 'processing',
    focus:
      'normalization (T = Sample / OB), proton-charge scaling, uncertainty propagation, ROI clipping, run combination, dark/air-region correction',
  },
  {
    key: 'tof',
    focus:
      'TOF binning, event/pulse reconstruction, resonance detection, statistics; off-by-one in bins, energy/wavelength conversions, numba vs pure-Python paths',
  },
  {
    key: 'pipelines',
    focus: 'MARS/VENUS end-to-end pipelines: metadata flow, output shapes/dims, ordering of corrections',
  },
  {
    key: 'exporters+filters',
    focus:
      'HDF5 (primary) + TIFF writers: round-trip correctness, dtype downcasts (float64 -> float32), units/attrs; gamma-spike filter',
  },
]

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
          why: { type: 'string', description: 'why this is a bug, grounded in the actual code' },
          fix: { type: 'string', description: 'concrete fix suggestion' },
        },
        required: ['severity', 'file', 'line', 'title', 'why', 'fix'],
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
    confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
    reason: { type: 'string' },
  },
  required: ['real', 'confidence', 'reason'],
}

const keyOf = (f) => `${f.file}:${f.line || '?'}:${(f.title || '').slice(0, 60)}`

phase('Find')
log(`Deep bug hunt scope: ${scope}`)

const claudeFinders = SUBSYSTEMS.map((s) => () =>
  agent(
    `Audit ${s.key} in src/neunorm for defects. Scope: ${scope}. Focus: ${s.focus}. ` +
      `Read the modules AND their tests; ground every finding in specific lines of the actual code. ` +
      `Report only defects you can trace to the source — prefer a few high-confidence findings over many speculative ones.`,
    { label: `find:${s.key}`, phase: 'Find', schema: FINDINGS_SCHEMA, agentType: 'bug-hunter' },
  ),
)

const codexFinder = () =>
  agent(
    `Provide an independent bug review using the Codex CLI when available. ` +
      `First check whether 'command -v codex' succeeds. If it does, run it read-only (no file edits): ` +
      `codex exec "Review ${scope} (a scipp-based neutron imaging library) for correctness, numerical, and edge-case bugs. ` +
      `For each finding give severity (P0/P1/P2/nit), file:line, why, and a fix. Do not modify files." ` +
      `Then translate Codex's output into the schema. If codex is NOT installed, return an empty findings list. Never edit files.`,
    { label: 'find:codex', phase: 'Find', schema: FINDINGS_SCHEMA },
  )

const finderResults = (await parallel([...claudeFinders, codexFinder])).filter(Boolean)

// Barrier is justified: dedup across ALL finders before the expensive verification
// pass, so the same issue is never verified more than once.
const seen = new Set()
const candidates = []
for (const r of finderResults) {
  for (const f of r.findings || []) {
    const k = keyOf(f)
    if (!seen.has(k)) {
      seen.add(k)
      candidates.push(f)
    }
  }
}
log(`${candidates.length} unique candidate findings to verify`)

phase('Verify')
const judged = await parallel(
  candidates.map((f) => () =>
    parallel(
      ['correctness', 'reproduce', 'edge-cases'].map((lens) => () =>
        agent(
          `You are a skeptical reviewer. Using the "${lens}" lens, try to REFUTE this finding by checking the actual code in the repo. ` +
            `Default to real=false unless the code clearly confirms the bug. Finding: ${JSON.stringify(f)}`,
          { label: `verify:${f.file}`, phase: 'Verify', schema: VERDICT_SCHEMA },
        ),
      ),
    ).then((verdicts) => {
      const votes = verdicts.filter(Boolean)
      const real = votes.filter((v) => v.real).length >= 2 // majority of 3 skeptics
      return { ...f, real, votes }
    }),
  ),
)

phase('Synthesize')
const confirmed = judged.filter((j) => j.real)
const rank = { P0: 0, P1: 1, P2: 2, nit: 3 }
confirmed.sort((a, b) => (rank[a.severity] ?? 9) - (rank[b.severity] ?? 9))
log(`Confirmed ${confirmed.length} / ${candidates.length} candidate findings`)

return {
  scope,
  confirmed,
  counts: {
    candidates: candidates.length,
    confirmed: confirmed.length,
    dropped: candidates.length - confirmed.length,
    bySeverity: confirmed.reduce((acc, f) => {
      acc[f.severity] = (acc[f.severity] || 0) + 1
      return acc
    }, {}),
  },
}
