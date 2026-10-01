# Mac validation provenance

Inspected 2026-10-01. This document is provenance only, never a runtime input.
The ECL seed-0 rho-1.0 pilot is LOCAL VALIDATION ONLY; its two completed online
trajectories and two scientific warmups must not count toward the cluster study.

The pilot's recorded execution environment (2026-10-01T04:46:38.323013+00:00)
reports CPython 3.10.6, macOS 26.6.2 / Darwin 25.6.0, arm64, 8 physical and
8 logical cores, CPU execution, deterministic algorithms, PyTorch 8 intra-op
and 8 inter-op threads. Hardware was the user-reported MacBook Air M1.
Recorded scientific versions: torch 1.12.0, NumPy 1.23.5, pandas 1.5.3,
scikit-learn 1.2.2. The pilot recorded no Git commit (null).

Current configured local virtual-environment package metadata agrees with those
recorded versions. Additional current metadata: pip 22.2.1, SciPy 1.10.1,
psutil 5.9.8, joblib 1.6.0, cloudpickle 3.1.2, threadpoolctl 3.7.0,
python-dateutil 2.9.0.post0, pytz 2026.3.post1, six 1.17.0,
typing_extensions 4.16.0. These additional versions were inspected after the
pilot; its environment JSON did not independently record them. torchdiffeq is
not installed and is not imported by v3.

Linux requirements preserve these package releases. PyTorch uses the official
1.12.0+cpu Linux x86_64 build instead of the Mac 1.12.0 arm64 wheel. This is an
explicit platform build difference, not an upgrade of the scientific release.
Different CPU libraries/platforms can produce numerical differences; the Mac
pilot is not a claim of bitwise Linux equivalence. Cluster results start fresh.

No usernames, hostnames, personal absolute paths, credentials, or scientific
results are embedded here. Do not transfer the local virtual environment.
