# Quickstart

Status: software release-candidate example; diagnostic only.

## Supported environment

Use 64-bit CPython 3.11 on Windows 11 for the primary target. Linux x86-64 with CPython 3.11 is the secondary portable target.

## Clean installation

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
python -m pip install ".[quickstart]"
std-tabular-diffusion quickstart --output artifacts/quickstart
~~~

If the Windows `py` launcher is unavailable, activate a Python 3.11 environment first and run `python -m venv .venv`, or replace `py -3.11` with the absolute path to a trusted CPython 3.11 executable. Stop if `python --version` does not report 3.11. On Linux, activate with `source .venv/bin/activate`; the remaining commands are unchanged.

The `.[quickstart]` form is for a source checkout. For a downloaded artifact, install `path/to/standardized_tabular_diffusion-0.1.0rc1-py3-none-any.whl[quickstart]` or the corresponding `.tar.gz[quickstart]`. After a future registry publication, use `standardized-tabular-diffusion[quickstart]`.

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

## Adult download-to-evaluation journey

The following commands exercise the user-owned workspace boundary, official Adult download, canonical preprocessing, real SMOTE training/generation, P3 evaluation, and Result Bundle validation. Run them from the parent directory in which you want `user-workspace` and `artifacts` to be created.

The first command below applies only in a source checkout. If you already installed a wheel or source archive with `[quickstart]` as described above, reuse that environment; the `quickstart` extra already contains the dependencies needed for this journey. Do not use the `.` form outside a source checkout.

~~~powershell
python -m pip install ".[quickstart,data]"
New-Item -ItemType Directory -Force user-workspace | Out-Null

std-tabular-diffusion materialize-dataset --dataset adult --workspace user-workspace
std-tabular-diffusion materialization-status --dataset adult --workspace user-workspace

std-tabular-diffusion create-diagnostic-dataset-profile `
  --dataset adult `
  --workspace user-workspace `
  --output user-workspace/adult-diagnostic-profile.json

std-tabular-diffusion example-config `
  --model smote `
  --dataset adult `
  --device cpu `
  --training-seed 17 `
  --generation-seed 17 `
  --num-samples 128 `
  --dataset-profile user-workspace/adult-diagnostic-profile.json `
  --protocol p3-validity `
  --output-dir artifacts/adult-smote-001 `
  --save-config user-workspace/adult-smote-001.json

std-tabular-diffusion run --config user-workspace/adult-smote-001.json --workspace user-workspace
std-tabular-diffusion validate-result --bundle artifacts/adult-smote-001/evaluation-result
~~~

The generated profile is intentionally `official_eligible: false`. It makes the repository's no-missing-model-input rule executable for a diagnostic P3 run, but it does not replace expert review of domains, cross-column rules, privacy roles, rights, or benchmark admission. Always use a new `--output-dir`; completed run artifacts are immutable.

## Evaluate your own decoded table

Use a reviewed Dataset Profile and a new output directory:

~~~powershell
std-tabular-diffusion evaluate-table --protocol p3-validity --reference real_train.csv --synthetic synthetic.csv --dataset-profile dataset-profile.json --output artifacts/my-result
std-tabular-diffusion validate-result --bundle artifacts/my-result
~~~

P4 and P5 additionally require `--real-test`. Never reuse an output directory: finalized evidence is immutable.

For missing values, first use the train-only preprocessing command described in [P3 Validity and Preprocessing](evaluation/P3_VALIDITY_AND_PREPROCESSING.md). For common failures, see [Troubleshooting](TROUBLESHOOTING.md).
