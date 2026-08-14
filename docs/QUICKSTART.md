# Quickstart

Status: software release-candidate example; diagnostic only.

## Supported environment

Use 64-bit CPython 3.11 on Windows 11 for the primary target. Linux x86-64 with CPython 3.11 is the secondary portable target.

## Clean installation

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ".[quickstart]"
std-tabular-diffusion quickstart --output artifacts/quickstart
~~~

On Linux, activate with `source .venv/bin/activate`; the remaining commands are unchanged.

The command uses a small artificial Apache-2.0 dataset included in the package. It downloads no dataset, uses no private input, and runs:

1. the unchanged official `imbalanced-learn==0.14.2` SMOTE implementation;
2. the central `p3-validity@0.3.0` evaluator;
3. Result Bundle validation; and
4. an immutable `partial-diagnostic` snapshot with no Official rank.

Inspect and independently validate the outputs:

~~~powershell
std-tabular-diffusion validate-result --bundle artifacts/quickstart/evaluation-result
std-tabular-diffusion validate-leaderboard --snapshot artifacts/quickstart/diagnostic-snapshot
~~~

`valid: true` means that schemas, checksums, cross-file invariants, and the selected diagnostic workflow passed. It does not mean SMOTE is a joint generative model, release-supported, benchmark-eligible, private, or scientifically superior.

## Evaluate your own decoded table

Use a reviewed Dataset Profile and a new output directory:

~~~powershell
std-tabular-diffusion evaluate-table --protocol p3-validity --reference real_train.csv --synthetic synthetic.csv --dataset-profile dataset-profile.json --output artifacts/my-result
std-tabular-diffusion validate-result --bundle artifacts/my-result
~~~

P4 and P5 additionally require `--real-test`. Never reuse an output directory: finalized evidence is immutable.

For missing values, first use the train-only preprocessing command described in [P3 Validity and Preprocessing](evaluation/P3_VALIDITY_AND_PREPROCESSING.md). For common failures, see [Troubleshooting](TROUBLESHOOTING.md).
