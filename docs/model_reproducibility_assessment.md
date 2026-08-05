# Model Reproducibility Assessment

Last reviewed: July 18, 2026

This document summarizes whether each model exposed by the standardized adapter registry appears to have a reliable, high-quality implementation that can be reasonably reproduced and run, along with the corresponding source repository and important usability notes.

## Model Categories

The current registry spans several distinct model groups. A practical categorization for benchmark reporting is:

### Traditional Generation

Statistical / classical:

- `smote`
- `bn`
- `arf`
- `tabsds`

VAE-based:

- `tvae`

GAN-based:

- `ctgan`
- `ctab-gan`
- `ctab-gan-plus`

Flow-based:

- `nflow`

Graph-based:

- `goggle`

### Diffusion Models

- `tabddpm`
- `tabsyn`
- `stasy`
- `codi`
- `tabdiff`

### LLM-Based Models

- `great`
- `realtabformer`
- `tabula`

### Autoregressive Specialized Models

- `tabularargn`

### Energy-Based Models

- `nrgboost`
- `tabebm`

## Registry-to-Category Map

| Model | Category | Subcategory / paradigm |
| --- | --- | --- |
| `arf` | Traditional generation | Tree-based / adversarial random forest |
| `bn` | Traditional generation | Statistical / Bayesian network |
| `codi` | Diffusion models | Conditional diffusion |
| `ctab-gan` | Traditional generation | GAN |
| `ctab-gan-plus` | Traditional generation | GAN |
| `ctgan` | Traditional generation | GAN |
| `goggle` | Traditional generation | Graph generative model |
| `great` | LLM-based models | Autoregressive transformer |
| `nflow` | Traditional generation | Normalizing flow |
| `nrgboost` | Energy-based models | Energy-based boosted trees |
| `realtabformer` | LLM-based models | Autoregressive / seq2seq transformer |
| `smote` | Traditional generation | Interpolation / oversampling baseline |
| `tabddpm` | Diffusion models | Score / denoising diffusion |
| `tabdiff` | Diffusion models | Multimodal diffusion |
| `tabebm` | Energy-based models | Class-conditional energy-based model |
| `tabularargn` | Autoregressive specialized models | Tabular autoregressive network |
| `tabula` | LLM-based models | Autoregressive language-model-based synthesis |
| `tabsds` | Traditional generation | Non-parametric rank-and-shuffle synthesis |
| `stasy` | Diffusion models | Score-based diffusion |
| `tabsyn` | Diffusion models | Latent diffusion |
| `tvae` | Traditional generation | VAE |

## Summary Table

| Model | Reliable runnable implementation? | Source repository | Important notes |
| --- | --- | --- | --- |
| `tabddpm` | Yes | [yandex-research/tab-ddpm](https://github.com/yandex-research/tab-ddpm) | Official implementation and widely reused. Reproducible, but upstream setup is dated and environment-sensitive. |
| `tabsyn` | Yes | [amazon-science/tabsyn](https://github.com/amazon-science/tabsyn) | Official implementation with a clear train/sample flow. Good research code, but dependency setup is somewhat brittle. |
| `tabdiff` | Yes | [MinkaiXu/TabDiff](https://github.com/minkaixu/tabdiff) | Official implementation with documented custom-dataset support. Good quality, but expects conda-style environment management and multiple evaluation environments. |
| `ctgan` | Yes | [sdv-dev/CTGAN](https://github.com/sdv-dev/CTGAN) | Best-maintained classic tabular generator in the set. Packaged, documented, tested, and actively released through the SDV ecosystem. |
| `tvae` | Yes | [sdv-dev/CTGAN](https://github.com/sdv-dev/CTGAN) | Shares the same maintained upstream as CTGAN. Easier to reproduce than most research-only baselines. |
| `smote` | Yes | [scikit-learn-contrib/imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) | Extremely reliable and actively maintained. Not a joint generative model, but a strong classical oversampling baseline. |
| `bn` | Yes through the locked canonical package; native parity validated for the declared recipe | [pgmpy/pgmpy](https://github.com/pgmpy/pgmpy) at `617cb48a`, PyPI `pgmpy==1.1.2` | The adapter uses unchanged official classes with an explicit quantile/BIC/BDeu recipe, includes isolated variables, and persists safe JSON graph/CPD state. All nine exact cases passed in retained Linux/Python 3.11 run `30967779298`. This is a declared-recipe claim, not a paper-native implementation claim. |
| `nflow` | Yes through a locked canonical library recipe; authoritative parity pending | [bayesiains/nflows](https://github.com/bayesiains/nflows) at `64b856c0`, PyPI `nflows==0.14` | The repository now declares the exact standardization, ordinal categorical representation, MAF architecture, optimizer, deterministic seed boundary, and decoding rule. It uses unchanged official classes and safe JSON/NumPy state instead of pickle. Because nflows is a toolkit rather than a paper-native tabular synthesizer, only exact package-plus-recipe parity can be claimed; the Linux/Python 3.11 evidence run is still pending. |
| `arf` | Yes via the method-author official Python package; native parity validated | [bips-hb/arfpy](https://github.com/bips-hb/arfpy), with the related [R method repository](https://github.com/bips-hb/arf) recorded separately | `arfpy==0.1.1` is authored in the same method-author organization by ARF authors. Its PyPI source distribution and runtime files are checksum-locked to commit `6f737baa`. The adapter calls official FORDE/FORGE, rejects missing values and the broken `oob=true` path, and stores safe JSON FORGE state without the forest or row-level training data. All nine exact official-package cases passed in retained Linux/Python 3.11 run `30964711614`; the claim does not establish R/Python cross-language equivalence. |
| `great` | Yes, but higher-maintenance | [tabularis-ai/be_great](https://github.com/tabularis-ai/be_great) | Maintained modern GReaT implementation. Better packaged than many paper repos, but still inherits HF/LLM runtime cost and sampling sensitivity. |
| `realtabformer` | Yes | [worldbank/REaLTabFormer](https://github.com/worldbank/REaLTabFormer) | Good engineering quality: tests, docs, packaging, and recent releases. Reproducible, but heavier than non-transformer baselines. |
| `tabula` | Partial / adapterized compatibility path | [zhao-zilong/Tabula](https://github.com/zhao-zilong/Tabula) | Official code exists, but the upstream is notebook-oriented. This repository now provides a local Transformers-based compatibility adapter instead of a direct wrapper around the original training flow, so reproduction should be treated as approximate rather than bit-for-bit. |
| `tabsds` | Partial / adapterized compatibility path | [echaibub/TabSDS](https://github.com/echaibub/TabSDS) | Official code exists and the method is lightweight, but this repository currently uses a local compatibility implementation inspired by the paper rather than the upstream code directly. |
| `tabularargn` | Yes, but optional-package dependent | [mostly-ai/mostlyai-engine](https://github.com/mostly-ai/mostlyai-engine) | Strong maintained implementation exists and the repository now exposes an adapter around it, but users must install `mostlyai-engine` separately and the path has not yet been smoke-validated here. |
| `nrgboost` | Yes, but relatively young | [Ajoo/nrgboost](https://github.com/Ajoo/nrgboost) | Official code and pip package exist. Runnable, but less battle-tested than CTGAN, TabDDPM, or imbalanced-learn style baselines. |
| `ctab-gan` | Native parity / checksum-locked official source | [Team-TUD/CTAB-GAN](https://github.com/Team-TUD/CTAB-GAN) | Seven selected files from pinned commit `73d4e315` replace the former semantic fork and retain Apache-2.0 attribution. The classification-only adapter preserves the official split and applies a documented non-semantic scikit-learn API bridge. All six exact parity cases passed in retained Linux run `30930939961`; Official Results and release gates remain pending. |
| `ctab-gan-plus` | Partial / checksum-locked official source | [Team-TUD/CTAB-GAN-Plus](https://github.com/Team-TUD/CTAB-GAN-Plus) | Five runtime files from pinned commit `6a6f901` are downloaded on demand and used without patches. All six native-parity cases passed on frozen Linux/Python 3.11, but the upstream repository declares no license, so redistribution and release support remain blocked. |
| `goggle` | Native parity for the method-author GCN core | [vanderschaarlab/GOGGLE](https://github.com/vanderschaarlab/GOGGLE) at `1a3d87ad` | The former materially modified TabSyn copy was removed. Eighteen official files are checksum-locked and acquired on demand under MIT without source patches. All nine exact binary/multiclass/regression and seed cases passed in retained Linux/Python 3.11 run `30945676747`. SAGE, heterogeneous decoding, Official Results, and release gates remain pending. |
| `stasy` | Native parity against the exact TabSyn benchmark snapshot; original-method equivalence blocked | The licensed runtime is pinned to [amazon-science/tabsyn](https://github.com/amazon-science/tabsyn). The separate [method-author repository](https://github.com/JayoungKim408/STaSy) is recorded but declares no license. | All 17 local STaSy files match the TabSyn subtree and the 30-file execution scope is fail-closed. All nine exact cases passed on Linux/Python 3.11 in run `30936275831`. The two repositories materially differ, so this evidence validates only the TabSyn snapshot. |
| `codi` | Exact TabSyn benchmark snapshot; original-method equivalence blocked | The licensed runtime is pinned to [amazon-science/tabsyn](https://github.com/amazon-science/tabsyn). The separate [method-author repository](https://github.com/ChaejeongLee/CoDi) is recorded but declares no license. | All 11 local CoDi files match the TabSyn subtree and the 24-file execution scope is fail-closed. The adapter confines both checkpoints to `output_dir` and supports exact requested rows. Five of ten method-author shared paths differ; all nine exact snapshot-parity cases passed in retained Linux/Python 3.11 run `30941940893`. |
| `tabebm` | Partial | Source linked by the paper/package: [andreimargeloiu/TabEBM](https://github.com/andreimargeloiu/TabEBM), package on [PyPI](https://pypi.org/project/tabebm/) | Credible method with a real package, but operationally gated by the modern TabPFN / Prior Labs ecosystem and external model access constraints. |

## Recommended Interpretation

Least surprising baselines to run:

- `ctgan`
- `tvae`
- `tabddpm`
- `tabsyn`
- `tabdiff`
- `smote`

Reasonably strong but higher-maintenance baselines:

- `realtabformer`
- `great`
- `tabula`
- `tabsds`
- `tabularargn`
- `bn`
- `nflow`
- `nrgboost`
- `ctab-gan`

Real implementations, but less reproduction-friendly:

- `ctab-gan-plus`
- `goggle`
- `stasy`
- `codi`
- `tabebm`
- `arf` if strict cross-language parity with the separate R implementation is required; the maintained target here is the official Python package

## Notes on Current Evidence

- The local adapter inventory and runtime notes are in:
  - [../standardized_tabular_diffusion/model_inventory.py](../standardized_tabular_diffusion/model_inventory.py)
  - [runtime_status.md](runtime_status.md)
- This assessment also checked current upstream repositories and package pages as of July 18, 2026.
- Example date-specific signals from upstream:
  - CTGAN latest release observed: February 13, 2026
  - REaLTabFormer latest release observed: January 4, 2026
  - pgmpy latest release observed: April 30, 2026
  - TabSyn README updates observed through June 20, 2024
  - TabDiff README updates observed through April 2025
