# Goggle Source, PyTorch Graph Backend, and Validation Record

Status: method-author GCN core native parity retained; Windows GPU real-function validation passed; formal DGL-oracle backend parity passed and retained<br>
Current protocol: `goggle-pytorch-graph-backend-parity-v2`<br>
Primary runtime: Windows 11, Python 3.11, PyTorch 2.8<br>
Independent graph oracle: Linux, Python 3.11, DGL 1.1.3

## Scope

This record separates two claims that must not be conflated:

1. The checksum-locked method-author Goggle GCN core was already validated with DGL under historical protocol `goggle-method-author-native-parity-v1`.
2. The current ordinary runtime keeps that Goggle source unchanged but replaces the unavailable Windows DGL dependency with a narrow, independently validated PyTorch graph backend.

The second item is a dependency compatibility reimplementation, not an unmodified official DGL runtime. It does not by itself admit Goggle to Official Results, the formal leaderboard, or release support.

## Authoritative Source

The reproduction target is the method-author implementation for the ICLR 2023 paper *GOGGLE: Generative Modelling for Tabular Data by Learning Relational Structure*:

- repository: `https://github.com/vanderschaarlab/GOGGLE`;
- commit: `1a3d87ad8a5dffe0f67f844e7b10f1f0dcef73e0`;
- repository tree: `2d6a54f6d6f4d156890bf4e035119dbb483a46d0`;
- `src/goggle` tree: `6dcaae801859f63e173537445548a50cd1f8625b`;
- license: MIT, copyright 2023 Tennison Liu; and
- locked archive SHA-256: `62dc6c98a2067d950513b4fe6343715f03a6a096990241fc6143b18fb56aaf65`.

The source manifest freezes 18 files. Materialization verifies the archive and every selected file before use. Source is stored in an ignored cache and is verified again after training and sampling. There are no upstream patch files and no official executable statements are edited.

The former `TabSyn-main/baselines/goggle` snapshot was materially different from the method-author source and has been retired. It is not used as official evidence.

## Why DGL Is Replaced

The supported environment is native Windows and Python 3.11. No official DGL wheel covers the selected Windows/Python/PyTorch runtime, and building a private DGL wheel would shift compiler and binary-maintenance risk to every user.

The validated Goggle GCN path executes only three DGL interfaces:

- construction of a directed homogeneous graph from ordered source and destination tensors;
- disjoint batching of repeated graphs; and
- weighted homogeneous `GraphConv`.

`standardized_tabular_diffusion.compat.goggle_torch_graph` implements only that audited surface. It is deliberately not a general DGL replacement. Ordinary Goggle installation and execution require neither DGL nor PyTorch Geometric. DGL remains installed only in the independent validation environment.

## Graph Semantics

Backend `standardized-goggle-torch-graph@1.0.0` targets DGL `GraphConv` 1.1.3 semantics used by Goggle:

- edge order and disjoint-batch node offsets are preserved;
- structural out-degree and in-degree normalization follows `none`, `left`, `right`, and `both` modes;
- edge weights scale messages but do not redefine structural degrees;
- multiplication occurs before aggregation when input width exceeds output width and afterward otherwise;
- weight initialization is Xavier uniform, bias initialization is zero, and state-dict keys remain `weight` and `bias`;
- bias and activation ordering matches DGL; and
- zero-in-degree nodes fail closed unless explicitly allowed.

Goggle's learned adjacency includes diagonal edges, so its supported path normally has no zero-in-degree nodes. Inputs outside this narrow contract raise a compatibility error instead of silently approximating DGL.

## Supported Model Path

The public adapter supports `decoder_arch="gcn"` only. `sage` and `het` are rejected before training because they execute different operators and have no validated compatibility implementation. Their import names are supplied only as fail-on-use placeholders so the unchanged upstream package can load without installing unused compiled extensions.

The official model, learned graph, encoder, GCN decoder structure, loss, optimizer alternation, seeded train/validation split, early stopping, and state-dict serialization remain in checksum-locked upstream code.

## Remaining Adapter Boundary

The adapter performs the following declared operations outside upstream source:

1. verifies source and artifact identities and confines the official relative checkpoint write to `output_dir`;
2. fits numerical standardization and deterministic categorical one-hot encoding on the real training split only;
3. reapplies the independent generation seed immediately before the unchanged stochastic sampler (the upstream constructor resets PyTorch to its training seed), passes the requested row count, applies the recorded inverse transform, and restores only explicitly declared integer columns with the repository-wide nearest-integer decoder;
4. supplies unused legacy Synthcity and heterogeneous-import names as fail-on-use placeholders; and
5. injects the recorded pure-PyTorch graph backend while importing the unchanged Goggle source.

Model metadata schema 2 records the backend ID, version, semantic target, source identity, transform, runtime configuration, and artifact digests. Sampling rejects legacy schema-1 metadata and any backend, source, configuration, or checkpoint mismatch. Models trained before this backend transition must be retrained.

## Data Contract

The adapter accepts classification and regression tables with any non-empty combination of numerical and categorical features, exactly one target column, canonical column order, finite numerical values, and no missing values. Missing values fail closed; users must first run the centralized train-split-fitted mean/mode imputer. The target remains part of the jointly synthesized vector.

Goggle models numerical values continuously. After the train-fitted inverse standardization, columns explicitly declared as integer by the canonical `DatasetSpec` are decoded with `numpy.rint` (ties to even) and `int64`, without clipping or test-set information. Continuous columns are untouched. The native inverse table and a per-column changed-row report are retained with each applicable sample run; the central validator still rejects invalid output rather than silently repairing it.

## Validation

Historical retained run [`30945676747`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30945676747) established exact native parity for the method-author GCN core under DGL 1.1.3. Its immutable evidence remains at `docs/evidence/goggle/native-parity-run-30945676747.json`; it does not validate the new dependency replacement.

Protocol v2 adds two independent layers:

1. graph-level comparison against DGL 1.1.3 for graph construction, batching, forward outputs, feature/edge/parameter gradients, state dictionaries, both multiplication branches, activation, and every normalization mode; and
2. nine end-to-end cases spanning binary classification, multiclass classification, regression, and seeds `0`, `19`, and `73`.

For every end-to-end case, the reference path uses unchanged Goggle plus DGL while the candidate path uses the same source plus the PyTorch backend. The gate requires exact checkpoint tensors, exact raw samples, exact final frames and CSV bytes, exact row/column contracts, valid metadata, and unchanged source files.

Formal workflow run [`32042446422`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/32042446422) passed on Linux/Python 3.11 with DGL 1.1.3 at commit `cf821a0dae803f523697c75888375feef9724145`. All five graph-level oracle cases and all nine end-to-end cases passed, including exact checkpoints, raw samples, final frames, and CSV bytes. The permanent record is [`pytorch-backend-parity-run-32042446422.json`](evidence/goggle/pytorch-backend-parity-run-32042446422.json), SHA-256 `77a4b3feb289703cbd78d2e03dd8338f2ae45c0eec0833fa6a0f25a769d54640`.

Windows GPU functionality passed at adapter commit `0a23a84`: PyTorch 2.8.0+cu128 on the declared RTX 5080 trained the 256-row Adult-derived fixture and produced valid, distinct 16-row outputs for generation seeds `17` and `29` while DGL and PyTorch Geometric were absent. Both runs preserved checkpoint bytes; the first output passed central `p3-validity` evaluation and finalized its Result Bundle. The retained record is [`windows-v2-real-function-0a23a84.json`](evidence/goggle/windows-v2-real-function-0a23a84.json). This demonstrates real functionality, not generation quality or benchmark eligibility.

## Usage

Materialize the locked source once:

```powershell
python -m standardized_tabular_diffusion.cli materialize-model-source --model goggle
python -m standardized_tabular_diffusion.cli model-source-status --model goggle
```

For the validated Windows GPU profile, install PyTorch from its official CUDA index first and then install the model extra:

```powershell
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[goggle]"
python -m standardized_tabular_diffusion.cli run --config configs/smoke/goggle-adult-smoke.json
```

The smoke preset is intentionally small and defaults to CPU. The Windows V2 validation plan overrides its device to CUDA.

Run the DGL oracle protocol only in the frozen Linux validation environment:

```bash
python -m standardized_tabular_diffusion.validation.goggle \
  --repo-root . \
  --output-dir /tmp/goggle-validation \
  --evidence-path /tmp/goggle-evidence.json
```

## Remaining Gates

The GCN core retains `native-parity-validated` provenance, the PyTorch backend now has retained formal Linux/DGL 1.1.3 v2 parity evidence, and the current Windows path is `minimal-real-passed`. This clears the dependency-reimplementation parity gate only. Goggle remains `experimental` and `unsupported`. Benchmark eligibility additionally requires approved datasets, the frozen central evaluation protocol, representative-scale resource qualification, and explicit admission. SAGE and heterogeneous decoding remain unsupported unless separately implemented and validated.
