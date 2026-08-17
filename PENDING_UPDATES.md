# Pending updates

Last successful maintenance: 2026-08-11 10:45 +02:00.

## Pending

- 2026-08-17: Harden the shared schema-2 manifest architecture for future
  upstream dependencies. Single-run resolution now fails when multiple
  pipeline configurations remain, accepts exact seed filters, and can bind a
  downstream computation to an upstream run's declared schema, identity/model
  configuration, pipeline/experiment parameters, and seeds while keeping paths
  and manifest IDs as provenance only. Affected files:
  `src/experiment_runs.py` and its focused contract tests. All 13 manifest tests
  passed. There are no current table-eligible schema-2 manifests or upstream
  selector, so this adds no migration or rerun beyond the already-required
  schema-2 experiment reruns. Deferred maintenance: document the capability if
  a future multi-stage producer is introduced.

- 2026-08-17: Replaced the duplicated two-inference evaluation with one pass
  that captures elementwise losses, stable individual/query/run IDs, and the
  optional example prediction together. `all_losses.pt` is now the sole loss
  artifact: each split stores flat `losses`, compact aligned `metadata`, and
  scalar `summaries`; per-user views are aggregated by ID without materialized
  tensor copies. Central/per-user merging, sklearn evaluation, dashboards, and
  table readers use the same contract. This is manifest schema 2; schema-1 runs
  require rerunning, although no table-eligible schema-1 run is currently
  recorded. Affected contracts/files: evaluation and generic batch metadata,
  PyTorch/sklearn loss collection, per-user merging, plotting/table readers,
  experiment result paths, manifest schema, README, cluster handoff, and focused
  tests. Checks passed: Python compileall for every changed Python module;
  `src/tests/test_evaluation.py`; `src/tests/test_sklearn.py`;
  `src/tests/test_results_table.py`; and all 11
  `src/tests/test_experiment_runs.py` lifecycle tests. Deferred maintenance:
  reconcile `latex/experiment_guideline.tex`, compile and inspect its PDF, and
  exercise one schema-2 `EXPERIMENT_MODE=test` cluster workflow. Required
  rerun: every desired schema-1 experiment configuration.

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
