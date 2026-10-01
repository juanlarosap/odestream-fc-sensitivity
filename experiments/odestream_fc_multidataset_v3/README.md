# Canonical ODEStream FC sensitivity framework

Implementation and static review only. No scientific work was executed while creating this package.

From the repository root:

```sh
python -m experiments.odestream_fc_multidataset_v3.run plan
python -m experiments.odestream_fc_multidataset_v3.run plan --dataset ETTh2 --seed 0 --rho 1.0
python -m experiments.odestream_fc_multidataset_v3.status
```

Future scientific commands (not executed during implementation):

```sh
python -m experiments.odestream_fc_multidataset_v3.run run --dataset ETTh2 --seed 0 --rho 1.0
python -m experiments.odestream_fc_multidataset_v3.run run
```

All commands accept `--output PATH`. Default: repository-relative
`outputs/odestream-fc-sensitivity-v3`. Use the same output directory for pilot and
full execution. `plan --json` prints the complete master graph and selected IDs.
Plan/status read only; they do not import model/engine modules. Use Python `-B`
or `PYTHONDONTWRITEBYTECODE=1` when filesystem-level bytecode writes must also be disabled.

## Frozen protocol

Dataset order ETTh2, ECL, Weather; seed order 0–4; within each seed Control Always,
then FC R50 rho 0.5, 0.75, 1.0, 1.25, 1.5. Exactly 90 scientific warmups and 90 online
trajectories (15 Control, 75 FC), plus three preparation warmups, three 128-step
calibration references, fifteen lambda derivations, and 75 validation bootstraps.
Filters select execution only. Control identities have no rho; FC identities do.
The full master graph and protocol hash never depend on filters.

ETTh2 uses seven inputs and OT; ECL uses only series_321 as input and target;
Weather uses the exact ordered 21 numerical inputs and OT. All use TRAIN-fitted
StandardScaler, lag 24 recorded observations, next recorded observation, and the
historical float32 linspace time grid. Weather's duplicate at row 19044 and
6000-second gap at row 21514 remain unchanged. Exact dataset SHA256 is required;
there are no downloads or repairs.

Model operations were extracted from root ODEVAE.py, NeuralODE.py and TIL.py.
PaperLoss-B/Adam/validation/bootstrap semantics derive from the finalized rolling
experiment and trajectory benchmark. Same-epsilon FC/EMA was extracted from the
finalized FC implementation. Runtime has no dependency on historical experiment
folders, checkpoints, reports or root model modules. Numerical definitions retain
historical operation order; comments in extracted model files are historical.

Preparation Control seed 0 is separate from scientific Control. Calibration uses
its validation-selected checkpoint, fresh Adam, checkpoint RNG and 128 TRAIN
lambda-zero updates with exact baseline mirror checks. Exclude only zero FC
norms; lambda = rho / median(G_cons/(G_base+1e-12)). Q95 is retained for each rho,
never used to select/reject grid points. No online outcomes change configuration.
Each scientific FC rho/seed has its own FC-regularized warmup. The online boundary
loads best VAE weights, preserves final validation RNG, initializes LSTM/fusion,
and creates fresh Adam and a new exact-copy teacher. No warmup deduplication.

R50 compares the current pre-update squared error with the prior 256-error linear
median, accepting ties. No additional label-delay queue; current error enters
FIFO after decision/update, even on skips. Skips consume student RNG but perform
no teacher forward, Adam or EMA. Same-epsilon teacher has stop-gradient; beta .99
EMA follows Adam. The initial FIFO uses the last 256 validation targets.

CPU only, deterministic algorithms, 8 intra-op and 8 inter-op threads, verified
against both historical support.runtime implementations. No CUDA/MPS selection.
Use the repository reproduction dependencies and a compatible Python environment;
no dependency installation or scientific portability test was performed here.

## Recovery and integrity

`run` owns an exclusive filesystem lock. Protocol, full plan, source manifest,
dataset manifests and numerical-stack compatibility validate before execution.
All existing master artifacts are reconciled before any new scientific work,
including artifacts outside the selected subset. COMPLETE checkpoints are checked
and skipped. Missing derived exports after a crash are reconstructed from their
complete checkpoint, without rerunning science. Completed lineage receipts pin
artifact hashes; corrupted or missing receipt-pinned exports fail closed.

Warmup commits every 10 batches and every epoch; online every 250 observations.
Signals request a safe-boundary checkpoint; abrupt termination replays only work
since the last durable checkpoint. Calibration commits every step. Bootstrap is
a frozen forward-only diagnostic; an interrupted, uncommitted bootstrap can be
repeated from its immutable warmup. No scientific training state is lost by doing
so. Checkpoints include full RNG, optimizer, student, teacher, histories and
positions. File and directory fsync plus atomic replace protect commits.

States: PENDING, RUNNING, INTERRUPTED, COMPLETE, FAILED. Invalid protocol/data/
source/checksum/lambda, nonfinite values, or other unexpected failures block the
whole experiment. There is no failed-job reset command or automatic restart.
Global and per-job failure markers preserve failures independently of state.json.
Manual investigation is required. `state.json` is a projection; checkpoints and
lineage are authoritative. An uncommitted RUNNING job is reconciled from artifacts.
A completed subset leaves the master PENDING/SUBSET_COMPLETE, e.g. 2/90 complete.

Protocol identity excludes host, absolute output path, filters, and machine
metadata. `execution_environment.json` and session records contain hostname,
CPU, cores, OS/kernel, Python/libraries, thread settings and Git commit (null if
unavailable). Changing hostname/path does not change scientific identity.
Changing a numerical library version during resume blocks compatibility checks.
The source manifest is independently frozen; code drift blocks resume.

Status is standard-library-only and read-only. UTC ISO-8601 timestamps have timezone
information. Warmup/online completion percentages each use denominator 90;
current warmup batch percentage is within its current epoch, not a prediction of
early stopping. Online progress is committed observations / eligible observations.
The dashboard labels stale/absent local processes without turning them into
scientific failures. Remote connection loss is not process-failure evidence.
ETA remains unavailable; no historical Mac times are used.

The online compute timer encloses batch retrieval and the complete historical
online step, including finite checks and built-in method diagnostics. Checkpoint,
trace, status, logging, initialization, warmup, calibration, bootstrap, and report
work remain outside. Replayed uncommitted compute is not added to the retained
scientific accumulator. Wall time is never promised to replay identically.

Reports are generated after all 90 trajectories validate. Seed-paired MSE change
is FC/Control - 1; runtime saving is 1 - FC/Control. Report mean, sample SD (ddof=1),
min/max and individual seeds. No winning rho is selected. Scientific reports
are never generated by plan, status, import, or static validation.
