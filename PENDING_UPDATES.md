# Pending updates

Last successful maintenance: 2026-08-11 10:45 +02:00.

## Pending

- 2026-08-26: Commented TiRex-2 out of the default frozen-foundation launcher
  while retaining its adapter and canonical alias, and extended every aligned
  foundation adapter's direct checkpoint discovery to the immediate parent
  `weights/` used by flat cluster checkouts. Existing per-dataset and
  per-checkpoint Slurm resolution already matched Adaptation's local, parent,
  and nested-workspace order. Affected contracts: foundation launcher default,
  aligned adapters, focused workflow regression, README, guidance, and
  experiment-guideline source. All four focused Slurm workflow tests and Git
  Bash syntax passed; SHA-256 checks confirmed all five adapter copies remain
  byte-identical across the three foundation projects. Real shared checkpoints
  remain a cluster check; maintenance rebuilt the guideline PDF and visually
  inspected all eight pages. No current completed result is reusable under the
  existing full-restart requirement; future foundation runs use four models.

- 2026-08-18: Returned TimeTensors to the thesis-wide manifest schema 1 after
  the full output/log reset and replaced the incompatible central/per-user MSE
  reduction. Every evaluated loss now records five population summaries from
  the same aligned tensor: `<metric>`, `std_<metric>`, `user_<metric>`,
  `std_user_<metric>`, and `w10_<metric>`. The per-user merger no longer
  overwrites the element-weighted metric, the central/per-user workflow emits
  separate comparable MSE and user-MSE tables, and the synthetic smoke records
  the same five MSE summaries. Affected contracts/files: manifest constant,
  evaluation/per-user aggregation, central/per-user Slurm workflow, synthetic
  smoke, focused tests, README, both LaTeX sources, and cluster handoff.
  Checks passed: evaluation and sklearn focused scripts; result-table script;
  all 13 manifest tests; all 3 Slurm workflow tests; and the complete synthetic
  smoke test. Outputs and logs still contain only their `.gitkeep` files.
  Deferred maintenance: compile and inspect both changed LaTeX PDFs. Required
  rerun: every desired TimeTensors experiment configuration and report.

- 2026-08-18: Deliberately reset all TimeTensors experiment state before a full
  restart. Removed 1,509 generated files from `outputs/` (including the legacy
  archive) and eight Slurm log files/four job pairs from `logs/`, while
  preserving both tracked `.gitkeep` placeholders. Replaced the cluster handoff
  with the empty-state and full-rerun contract. Verification found exactly one
  `.gitkeep` in each directory and no remaining artifact or log. No code,
  configuration, README, or LaTeX source changed. Before any cluster submission,
  resolve the thesis-wide schema-version inconsistency and make the central and
  per-user MSE table use the same weighting. Required rerun: every desired
  TimeTensors experiment configuration and report.

- 2026-08-17: Replaced the duplicated two-inference evaluation with one pass
  that captures elementwise losses, stable individual/query/run IDs, and the
  optional example prediction together. `all_losses.pt` is now the sole loss
  artifact: each split stores flat `losses`, compact aligned `metadata`, and
  scalar `summaries`; per-user views are aggregated by ID without materialized
  tensor copies. Central/per-user merging, sklearn evaluation, dashboards, and
  table readers use the same contract. This initially used manifest schema 2,
  but that version choice was superseded after the full reset by the schema-1
  correction above. Affected contracts/files: evaluation and generic batch metadata,
  PyTorch/sklearn loss collection, per-user merging, plotting/table readers,
  experiment result paths, manifest schema, README, cluster handoff, and focused
  tests. Checks passed: Python compileall for every changed Python module;
  `src/tests/test_evaluation.py`; `src/tests/test_sklearn.py`;
  `src/tests/test_results_table.py`; and all 11
  `src/tests/test_experiment_runs.py` lifecycle tests. Deferred maintenance:
  reconcile `latex/experiment_guideline.tex`, compile and inspect its PDF, and
  exercise one current-contract `EXPERIMENT_MODE=test` cluster workflow.
  Required rerun: every desired experiment configuration.

- 2026-08-17: Simplify `publish_job.sh`: a numeric job ID now selects only its
  exact stdout/stderr pair, while an omitted ID stages the `logs/` and
  lightweight `outputs/` parent trees directly. Publisher, focused contract
  test, README, and shared guidance changed. The project publisher contract
  test and Git Bash syntax passed, and all nine copies have matching SHA-256
  hashes. No scientific rerun or artifact migration is required. Deferred
  maintenance: reconcile and render the experiment guideline; retain the
  existing real-cluster publisher integration check.

- 2026-08-16: Adopt the thesis-standard `publish_job.sh`: source the proxy and
  fast-forward pull `origin/main` before artifact selection, staging, or commit,
  then publish only the lightweight selected paths. Affected contracts:
  publisher, focused contract test, README, and shared experiment guidance.
  Checks passed: Bash syntax for all nine standard copies, matching SHA-256
  hashes, and the TimeTensors publisher contract test. No scientific rerun or
  artifact migration is required. Deferred maintenance: reconcile
  `latex/experiment_guideline.tex` and exercise one real cluster publish with a
  remote update present.

- 2026-08-12: Synchronize Adaptation's terminal lifecycle: remove automatic
  publisher submission, add the manual root `publish_job.sh`, restrict overall
  manifests to `not_run|running|interrupted|completed`, and allow tables to
  consume seed-ready artifacts only from their own active launch. Affected
  files/contracts: manifest helper, benchmark runner, table reader, publisher
  files, focused tests, README, and parent experiment guidance. Checks passed:
  13 focused lifecycle/publisher/Slurm/table tests and Bash syntax for the
  runner and manual publisher. No scientific rerun or artifact migration is
  required. Deferred maintenance: reconcile and render
  `latex/experiment_guideline.tex`; cluster-check one successful and one
  failed/cancelled launch, then run the manual publisher once.
  Maintenance 2026-08-13: direct inspection confirmed the four-state overall
  manifest, seed-only `ready`, same-launch table selection, and final exit
  promotion. Config-default, dataloader, model, sklearn, and complete synthetic
  family/seed coverage tests passed in the shared thesis runtime. The README
  and guideline now describe the exact lifecycle and manual publisher. Three
  pdfLaTeX passes stabilized cross-references without warnings, and all seven
  rendered pages passed visual inspection. The previously successful
  lifecycle/publisher/Slurm/table checks were not repeated because these five
  checks cover complementary project boundaries. Remaining blocker: observe
  one successful and one failed/cancelled cluster launch, then run
  `publish_job.sh` once.

- 2026-08-13: Complete every successful TimeTensors configuration immediately,
  preserve it across later workflow failure, interrupt only unfinished runs,
  and retain per-seed artifact lists. Affected contracts: shared manifest
  helper, runner, tests, README, and experiment guideline. Checks passed: 11
  lifecycle tests, publisher and three workflow contracts, Python AST parsing,
  Bash syntax, clean LaTeX compilation, and visual inspection of all seven PDF
  pages. No artifact migration, scientific rerun, or schema bump is required.
  Remaining cluster work: exercise successful and failed/cancelled launches and
  run the manual publisher once.

Maintenance 2026-08-16: no source, configuration, artifact, documentation, or
cluster-handoff file changed after the previous pass. Direct inspection again
found completed-only reuse, same-launch ready table selection, and the manual
publisher contract. The already successful configuration, dataloader, model,
sklearn, synthetic-family, and PDF checks were not repeated because there is no
changed integration boundary. Live successful/failed launch observations and
one manual publisher run remain the sole blocker; no scientific rerun is
required.

Maintenance 2026-08-17: direct inspection found no new source, artifact, or
cluster-status change and reconfirmed completed-only reuse and same-launch ready
table selection. The README was current; the experiment guideline was
reconciled with the canonical proxy-first, fast-forward-pull publisher. Bash
syntax passed for all nine byte-identical copies. Three pdfLaTeX passes
completed with a clean log, and all seven rendered guideline pages passed
visual inspection. The prior configuration, model, synthetic-family, and
lifecycle checks were not repeated because those boundaries did not change.
Live successful/failed launch observations and one real publisher run remain
the blockers; no scientific rerun is required.

Maintenance 2026-08-18: direct code inspection confirmed the single-pass,
row-aligned `all_losses.pt` contract, the then-current schema-2 choice, shared
fail-closed dependency support, and the absence of table-eligible runs. The prior 13 focused
manifest tests close the standalone dependency-helper entry; the evaluation
evaluation entry remains open for cluster execution. As complementary coverage,
the complete seven-family/two-seed synthetic smoke test passed (1 test), and
the Slurm workflow contract passed (3 tests). Git Bash syntax passed for all
nine byte-identical publishers. The README and cluster handoff were current;
the experiment guideline was reconciled with the then-current schema choice, the sole aligned loss
artifact, dependency signatures, and exact-log publication.
Three pdfLaTeX passes completed with a clean log, and all seven rendered
guideline pages passed visual inspection. Later on 2026-08-18, the full local
reset and schema-1/metric correction above superseded that schema choice and
made every desired configuration a required rerun. Remaining work is one
current-contract `EXPERIMENT_MODE=test` cluster workflow, live success/failure
lifecycle observation, and one real publisher integration.

Maintenance 2026-08-19: direct inspection confirmed `schema_version: 1`, the
single aligned `all_losses.pt` payload, the five element/user/dispersion/tail
summaries for every evaluated metric, separate comparable `mse` and
`user_mse` central/per-user tables, and exactly one `.gitkeep` file in each of
`outputs/` and `logs/`. Complementary `test_config_defaults.py` and
`test_dataloaders.py` checks passed in the shared thesis runtime. The focused
evaluation, sklearn, result-table, manifest, Slurm, and complete synthetic
smokes were not repeated because they had already passed for these exact
changed boundaries. The README's deleted-archive wording was reconciled. The
guideline and reset-aware executive summary each compiled cleanly after three
pdfLaTeX passes; all seven guideline pages and both summary pages passed visual
inspection. Every desired experiment configuration still requires rerunning,
starting with `EXPERIMENT_MODE=test`; live success/failure lifecycle observation
and one real publisher run also remain external blockers.

Maintenance 2026-08-20: direct timestamp, repository, cluster-handoff, and
artifact inspection found no change after the previous pass and reconfirmed
that `outputs/` and `logs/` contain only their placeholders. The README,
guideline, and reset-aware executive summary remain current, and the publisher
remains byte-identical across all nine projects at SHA-256
`0A9E87E51517B9F5816BB92CDE726B9E383AB6B8A70DC251FEF429BF7B53B45C`.
Scientific, manifest, Slurm, synthetic, Bash-syntax, and PDF checks were
deliberately skipped because their inputs and integration boundaries are
unchanged. Every desired configuration still requires rerunning, beginning
with `EXPERIMENT_MODE=test`; live lifecycle observation and one real publisher
run remain external blockers.

Maintenance 2026-08-23: direct inspection confirmed the shared nested
selection and deterministic latest-run behavior, and the helper plus focused
test file are byte-identical to the other four maintained copies. The
complementary synthetic `src/tests/test_results_table.py` consumer passed in
the shared thesis runtime, covering current manifests through table rendering.
README selection documentation and the full-reset handoff remain current; no
LaTeX, result claim, or migration changed. The selector entry is resolved and
adds no rerun beyond the already-required full restart; cluster lifecycle and
publisher checks remain pending.

Maintenance 2026-08-24: direct package, import, notebook, configuration,
Slurm, test, placeholder, and full-restart handoff inspection confirmed the
explicit owners, clean proposal/external-model boundaries, and absence of
compatibility paths. As complementary coverage, all relocated packages plus
the experiment and report modules imported together in the shared thesis
runtime. A Hydra `--help` attempt was inapplicable because Hydra is absent and
the fallback stopped before training on the missing tensor payload; the empty
automation-created `outputs/manual_debug/` directory was removed. README and
both LaTeX documents remain current, so the reorganization and guidance
entries are resolved with no additional rerun. Every desired cluster
configuration, lifecycle observation, and publisher check remains pending.

## 2026-08-24 — Shared foundation evaluation and cohesive data owners

- Behavior and affected contracts: replaced the monolithic data owner with
  core, sampling, frames, I/O, splits, statistics, and loaders; added TIME
  preparation and an evaluation-only eighth workflow for Chronos-2,
  Chronos-Bolt, TS-ICL, TiRex-2, and TabPFN-TS; retained the richer local
  chronos and tabpfn specializations under distinct names; and aligned
  PatchTST/DLinear source packages and the five shared adapter contracts.
- Focused checks and outcomes: Python compilation passed; model, data-loader,
  evaluation, TIME-preparation, package-layout, and Slurm contract checks
  passed; Git Bash accepted the changed workflow files; TOML parsing passed;
  SHA-256 comparison found one hash per shared TIME/adapter/source snapshot.
- Deferred integration: uv is unavailable in the documented local runtime and
  user-managed environment changes are out of scope, so uv.lock remains stale
  against the new exact dependency pins. Checkpoint-backed inference and the
  new Slurm front remain cluster checks.
- README/LaTeX and reruns: README documents the new owners, shared adapters,
  provenance, TIME preparation, and eighth workflow. Reconcile and render the
  experiment guideline during maintenance. The already-required complete
  experiment restart covers every changed trained/external-model result.

Maintenance 2026-08-25: direct owner, package, workflow, placeholder, README,
guideline, summary, and handoff inspection confirmed that outputs and logs still
contain only placeholders and that the full restart now includes the frozen
foundation family. The complementary `src/tests/test_synthetic_smoke.py` check
passed (1 test), exercising every expected method in the seven trainable/control
families across both required synthetic seeds. The README and reset-aware
summary were already current. The guideline now specifies the eighth family,
TIME preparation, five official adapters, provenance, and current artifact
identity; final pdfLaTeX passes produced a clean seven-page PDF and all pages
passed visual inspection. `uv` remains unavailable and no environment was
changed. Refreshing the lock, checkpoint-backed foundation smoke, the complete
cluster restart, lifecycle observations, and publisher check remain pending.

## 2026-08-25 — One canonical alias per foundation model

- Behavior and affected contracts: reduced the foundation registry to exactly
  `chronos2`, `chronos_bolt`, `ts_icl`, `tirex2`, and `tabpfn_ts`; removed the
  distinct `chronos` and `tabpfn` implementations; routed reference and
  central/per-user Chronos-2 cases through `chronos2`; and reject historical,
  case-variant, mismatched name/path, and direct-import bypasses. Project
  normalization and covariate augmentation remain outer `TimeTensorModel`
  configuration rather than model identities. Chronos-Bolt now rejects both
  structured and named covariates.
- Focused checks completed: `src/tests/test_models.py` and
  `src/tests/test_slurm_workflow.py` passed through their direct entry points;
  Python AST parsing, changed-workflow Bash syntax, exact five-alias parity,
  and cross-project SHA-256 parity for all five basic adapters passed. `pytest`
  was unavailable in the prepared runtime and no environment was changed.
- Deferred integration: checkpoint-backed covariate calls and the foundation
  Slurm front remain cluster checks. The user-managed `uv` environment and
  lockfile were not changed.
- README/LaTeX and reruns: README and the guideline source now document the
  single-alias wrapper contract. Re-render the guideline during maintenance;
  no executive-summary claim changed. The existing complete-restart requirement
  covers every result formerly identified by `chronos`, `chronos-bolt`, or
  another removed spelling.

Maintenance 2026-08-26: direct workflow, model registry, report consumer,
placeholder, README, guideline, summary, and handoff inspection confirmed the
single-task launches and empty active artifact state. The complementary
`src/tests/test_results_table.py` consumer passed without training. The
guideline was newer than its PDF because of the canonical-alias update; three
final pdfLaTeX passes produced a clean seven-page PDF and every page passed
visual inspection. The assertion-only and empty archive-storage entries are
resolved. The complete restart, managed lock refresh, checkpoint-backed
foundation workflow, lifecycle observations, and publisher check remain
pending.
