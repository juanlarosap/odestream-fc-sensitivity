# ODEStream ablation v1: the two missing conditions

This sibling package executes exactly **30 new online trajectories**: ETTh2,
ECL, Weather × seeds 0–4 × `r50_only`, `fc_only`. Each has its own warmup.
It never executes Control, R50+FC, sensitivity, corruption, or calibration.
All files in `../odestream_fc_multidataset_v3/` remain unchanged.

## Frozen treatment

| Condition | FC | Online gate | Warmup | Bootstrap | Coefficient |
|---|---|---|---|---|---|
| `r50_only` | OFF | R50 | Base ODE-VAE | Final 256 validation targets | None |
| `fc_only` | ON | Always | FC-aware ODE-VAE | None; no FIFO | Existing dataset v3 rho=1.0 |

The FC factor includes **warmup and online FC**. This is not an online-only FC
experiment. No checkpoint is shared between the two warmup families.

The package directly reuses v3 data loading/scaling/windows, model and solver,
base loss, FC/EMA, RNG, validation, warmup, transition, bootstrap, Gate,
`online_step`, and `timed_observation`. The online loop only separates gate
selection from the internal base/FC model variant. All timed operations,
including rejected-step base losses and FC diagnostics, remain in v3 code.

Warmup: chronological full batches of 64 (drop incomplete batch), Adam 0.001,
100-epoch cap, patience 10, stochastic validation MSE. Online transition loads
best VAE weights but retains final-validation RNG, constructs fresh LSTM/fusion
and Adam, and resets the FC teacher to an exact copy of the complete student.
CPU only; deterministic algorithms; 8 intra-op and 8 inter-op threads. No seed
offsets or data workers. Standardized-target MSE/MAE and online timing match v3.

## Package layout

- `config.py`: independent public condition/FC/gate identities and frozen plan.
- `engine.py`: minimal explicit-gate loop, validation, and result adaptation.
- `provenance.py`: explicit reused source closure and read-only calibration checks.
- `run.py`: CLI, locking, resumable stages, immutable identities and receipts.
- `reporting.py`: checkpoint-validated standalone five-seed summaries.
- `test_protocol.py`: small synthetic tests; no dataset scientific trajectories.

The separate provenance module keeps artifact verification out of the scientific
loop. This is the only addition to the seven-file proposed package layout.

## Calibration and identity

Default authoritative root: `outputs/odestream-fc-sensitivity-v3`.
Expected cluster absolute root:
`/data/juanca/odestream-fc-sensitivity/outputs/odestream-fc-sensitivity-v3`.
Override with `--v3-output`. FC-only requires, for each selected dataset:

- original `protocol.json`, `plan.json`, `sources.json`, execution environment;
- `datasets/<dataset>/calibration/reference.json` and its lineage receipt;
- `datasets/<dataset>/calibration/rho_1.0.json` and its lineage receipt;
- referenced preparation `control_seed_0/state.pt` and lineage receipt.

Verification checks exact clean-v3 protocol/plan, dataset/node/source identities,
parent and receipt hashes, the 128 seed-0 lambda-zero gradient observations,
positive-FC-gradient median, and v3's exact `derive(reference, 1.0)` fields.
The preparation checkpoint is read as bytes for its hash, not used for a new
scientific warmup. No Control/R50+FC result or online checkpoint is imported.
No old manifest or artifact is changed. No calibration function is executed.

The original source inventory may contain additional unrelated files. Only the
explicit reused v3 module hashes must match. New identity records the ablation
Python files plus this explicit import closure; unrelated v3 corruption/CLI/status
files are excluded. Dataset hashes, referenced artifact hashes, and environment
are recorded separately. `calibration_reference.json` in the ablation output is
an identity/coefficient record, not a copy of old scientific results/checkpoints.

R50-only needs no v3 calibration directory. An invocation selecting only R50-only
does not load FC coefficients, including when resuming a shared output root.
Unselected completed work is checked by receipt hashes. Full scientific artifact
validation occurs when selected and before aggregation.

`verify` is read-only and checks the active numerical environment against the
referenced calibration environment. `run` repeats these checks before FC work.
It does not certify that a directory is authoritative from its name alone: use
the finalized cluster directory, never the local Mac pilot. Compare CPU/OS and
resource allocation printed by `verify` before measuring new authoritative times.
Library/thread/device mismatches fail; matching CPU allocation and low contention
remain operator requirements for historical runtime comparability.

## Local validation (no full scientific jobs)

Activate the existing PyCharm scientific environment (do not upgrade dependencies).
From repository root:

```sh
source .venv/bin/activate
python -B -m experiments.odestream_fc_ablation_v1.run plan
python -B -m experiments.odestream_fc_ablation_v1.run plan --json
python -B -m unittest experiments.odestream_fc_ablation_v1.test_protocol -v
python -B -m experiments.odestream_fc_ablation_v1.run verify \
  --conditions r50_only
python -B -m experiments.odestream_fc_ablation_v1.run verify \
  --v3-output outputs/odestream-fc-sensitivity-v3
```

The last command intentionally fails if any required calibration is absent. The
local audited output root contains only ECL seed-0 pilot calibration; it cannot
verify the full authoritative study. To check that pilot's artifact structure only:

```sh
python -B -m experiments.odestream_fc_ablation_v1.run verify \
  --dataset ECL --conditions fc_only \
  --v3-output outputs/odestream-fc-sensitivity-v3
```

Unit tests use temporary directories and small synthetic windows. They do not
write scientific outputs under `outputs/` or inside v3. Python `-B` prevents
bytecode writes. The plan always has 30 online jobs; filters select subsets
without changing its identity. There is no `--rho` option.

**A filtered `run` is still a full scientific dataset trajectory, not a tiny
smoke test.** Do not run it on the Mac as part of routine validation. If a local
full pilot is separately authorized later, isolate and label it explicitly:

```sh
python -B -m experiments.odestream_fc_ablation_v1.run run \
  --dataset ETTh2 --seed 0 --conditions r50_only fc_only \
  --v3-output outputs/odestream-fc-sensitivity-v3 \
  --output outputs/odestream-fc-ablation-v1-local --non-authoritative
```

This requires valid ETTh2 calibration and a compatible environment; the local
ECL pilot does not supply it. NON-AUTHORITATIVE status is frozen and propagated
to every result/report/completion manifest. Never merge local pilot outputs
into cluster outputs. There is deliberately no shortened scientific-run flag.

## Git transfer

All new source files are in this folder. Existing datasets stay in `data/`;
old outputs remain on the cluster. No checkpoint transfer is needed.

```sh
git status --short
git diff -- experiments/odestream_fc_multidataset_v3
git add experiments/odestream_fc_ablation_v1
git diff --cached --stat
git commit -m "Add frozen R50-only and FC-only ablation"
git push
```

`/outputs/` is already ignored by the repository. Do not add it or any datasets.
Review staged changes to avoid including unrelated work.

## Cluster verification and execution

Use the existing v3 scientific environment, matching the original hardware and
resource allocation. Do not run the old fresh-install `cluster_preflight.sh`:
that script deliberately rejects an existing v3 output directory.

```sh
cd /data/juanca/odestream-fc-sensitivity
git pull --ff-only
source .venv/bin/activate
python -B -m experiments.odestream_fc_ablation_v1.run plan
python -B -m experiments.odestream_fc_ablation_v1.run verify \
  --v3-output outputs/odestream-fc-sensitivity-v3
python -B -m experiments.odestream_fc_ablation_v1.run run \
  --conditions r50_only fc_only \
  --v3-output outputs/odestream-fc-sensitivity-v3 \
  --output outputs/odestream-fc-ablation-v1
```

Run serially in a suitable cluster allocation or persistent terminal. Reissue the
same `run` command after interruption. A single exclusive lock protects the output
root. Warmup checkpoints follow v3 (every 10 batches and epoch boundaries); online
checkpoints follow v3 (every 250 observations). Signals checkpoint at a safe
boundary. Uncommitted work may replay; its old wall time is not counted twice.
Full RNG/student/teacher/Adam/FIFO/trace state is restored. Completed work is
validated and skipped. Invalid artifacts or unexpected failures create a durable
FAILED marker; there is no automatic reset, deletion, or scientific replay.

```sh
python -B -m experiments.odestream_fc_ablation_v1.run status \
  --output outputs/odestream-fc-ablation-v1
python -B -m experiments.odestream_fc_ablation_v1.reporting \
  --output outputs/odestream-fc-ablation-v1
```

Status reports committed progress only; it does not assert that a recorded PID
is alive. Reporting takes the same lock and requires all 30 completed trajectories.
It also runs automatically at completion. Reports can be regenerated identically
without rerunning science. Completed checksum-pinned artifacts fail closed on drift.

## Output and reporting

```text
outputs/odestream-fc-ablation-v1/
  protocol.json, plan.json, sources.json, execution_identity.json
  execution_environment.json, execution_sessions/, state.json, runner.lock
  datasets/<dataset>/
    dataset_manifest.json
    calibration_reference.json                 # FC provenance only
    r50_only/seed_<0..4>/
      warmup/state.pt
      bootstrap/state.pt
      online/{state.pt,result.json,trace.csv.gz}
    fc_only/seed_<0..4>/
      warmup/state.pt
      online/{state.pt,result.json,trace.csv.gz}
  reports/{aggregate.json,summary.csv,REPORT.md}
  completion_manifest.json
```

Each stage has a sibling `.lineage.json` receipt. Results use public `condition`,
`fc_enabled`, and `gate_mode`; no public base/FC `variant` ambiguity. The accepted
update-count field is named `updates`, matching v3. FC-only reports rho=1.0/lambda;
R50-only has neither. The report includes MSE, MAE, online compute seconds, update
fraction, accepted updates, and eligible observations: mean, sample SD (`ddof=1`),
min, max, and individual seed values ordered 0–4. No Control-relative metrics are
calculated. Online time excludes warmup/calibration/bootstrap and persistence,
exactly as v3. Comparing R50-only with R50+FC later measures full treatment cost,
including any FC-induced change in acceptance frequency.
