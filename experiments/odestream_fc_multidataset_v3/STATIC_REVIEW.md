# Implementation/static validation record

No warmup, calibration, scientific model construction, forward/backward, pilot,
stream, or scientific trajectory was executed. No scientific outputs were created.

## Static evidence

- Every Python source compiles in memory (no pyc writes).
- Pure configuration/CLI checks: 3 datasets, 5 seeds, 5 rho values; 15 Control +
  75 FC online jobs; 90 distinct scientific warmups. Complete DAG: 276 nodes,
  including 3 preparation warmups, 3 references, 15 lambdas and 75 bootstraps.
- Pilot closure: preparation warmup, calibration reference, rho-1 lambda,
  scientific Control warmup/Always, scientific FC warmup/bootstrap/R50: 8 nodes.
  Exactly 2 scientific warmups and 2 online jobs. Master identity remains fixed.
- Pilot IDs are members of the full plan. Removing these two online IDs leaves 88.
  No scenarios or scientific artifacts were instantiated to check this property.
- All modules imported with nn.Module construction forbidden; Torch RNG unchanged.
- Plan and status dispatched under filesystem-write denial with no scientific
  library imports. Run syntax was parsed, never dispatched.
- ISO-8601 timestamp includes UTC offset; percentages checked at 0, 2, 43 and 90.
- Actual ETTh2, ECL and Weather bytes matched all expected SHA256 values.
- AST equality verified for extracted scientific loss, Adam, numerical checks,
  ODE solver/adjoint/vector field, encoder/decoder/VAE, LSTM and fusion classes.
  FC AST matches after explicit dependency-name substitution. The memory profiler
  decorator was removed, as historical runners already bypassed it. NeuralODE's
  unused default time tensor is now constructed on call, preventing import-time
  tensor construction; supplied scientific time-grid behavior is unchanged.
- Historical preservation comparison covered 2,257 files: identical file set,
  sizes and modification timestamps; source/Markdown/JSON content hashes match.
  Binary checkpoint payloads were not deserialized or rehashed for this protection
  check. No command wrote inside any historical folder.

## Source lineage

| Canonical module | Authoritative source |
|---|---|
| models.py, neural_ode.py | Root ODEVAE.py, TIL.py, NeuralODE.py |
| losses.py | odestream_original_loss_rolling/run_experiment.py |
| fc.py | trajectory_stabilization/functional_consistency/functional_consistency.py |
| model_adapter.py, engine.py | trajectory_stabilization/benchmark/protocol.py and engine.py; historical validation/bootstrap |
| datasets.py | finalized ETTh2 loader, multidataset_v2 ECL adapter, Weather adapter |
| runner.py, checkpointing.py | shared benchmark state contracts plus Weather durable workflow concepts |
| reporting.py | finalized paired seed formulas, sample SD and checkpoint-backed metrics |

These are extraction/design references, not runtime dependencies. No historical
folder is imported or required at runtime.

## Static resume control-flow review (not executed)

A. Fresh repository: freeze full protocol/plan/source identity, verify all data,
record environment, build dataset manifests, reconcile absent artifacts as PENDING,
then execute selected dependencies in frozen order. Missing data blocks; no download.

B. Successful ETTh2 seed-0 rho-1 pilot: reconciliation loads and validates its
preparation, reference/lambda, two warmups, bootstrap and two online checkpoints,
receipts and exports. Complete jobs are skipped. Default full run derives missing
rho lambdas and continues remaining members; 2/90 complete, 88 remain. Control
has no rho and cannot be duplicated by another rho filter.

C. Halfway through FC warmup: valid partial checkpoint restores student, teacher,
Adam, epoch, next batch, best snapshot, stopping state, history, totals and RNG.
Committed batches are skipped. Only work after the last checkpoint replays.
A graceful signal commits at the next safe batch boundary; abrupt termination
uses the last 10-batch/epoch commit. A corrupt checkpoint fails, never restarts.

D. Online observation 11750: when that periodic checkpoint is committed,
next_index=11750 means 11750 targets have been processed; resume begins with
zero-based index 11750 (the 11751st target). Student/teacher/Adam/RNG/FIFO/update
count/trace/metric sums/time restore together. If killed before the atomic commit,
resume from the preceding durable checkpoint. Complete warmup/bootstrap skip.

E. 43 online jobs complete: reconcile all existing master artifacts before new
science, skip the 43 complete trajectories, and resume/execute remaining jobs
in fixed order. 47 remain (47.78% complete, 52.22% remaining). Warmup counts are
computed separately, not inferred from online counts. Any complete or partial
artifact outside an execution filter is still validated.

F. NaN/FAILED: numerical finite checks raise, preserve last committed checkpoint,
write durable global/per-job failure markers and FAILED state, then stop. Any later
run blocks before scientific work even when filters omit the failed trajectory.
Missing state.json does not erase a durable failure marker. No reset/delete policy
is automated. Investigation is required.

## Recovery ordering and remaining limits

Artifacts commit before the global projection. Complete checkpoints without a
receipt/export after a crash are validated and exports recovered without rerunning
science. A receipt already present pins hashes: corruption/missing pinned outputs
fails closed. An interrupted bootstrap can repeat its uncommitted frozen forwards
from its validated warmup; it performs no optimization and restores initial RNG.
Calibration resumes stepwise, including the case of 128 completed diagnostic steps
with the final reference JSON not yet committed.

No numerical execution parity or actual crash-resume test was performed. Those
remain future authorized validation, not claims made by this static review.
Cluster dependencies/filesystem behavior have not been tested on Linux hardware.
Git commit is recorded as null when Git metadata is unavailable; this workspace
has no usable Git repository metadata. Status ETA deliberately remains unavailable.
