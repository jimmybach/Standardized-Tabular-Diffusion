# Model Reproducibility Assessment

Last reviewed: July 16, 2026

This document summarizes whether each model exposed by the standardized adapter registry appears to have a reliable, high-quality implementation that can be reasonably reproduced and run, along with the corresponding source repository and important usability notes.

## Summary Table

| Model | Reliable runnable implementation? | Source repository | Important notes |
| --- | --- | --- | --- |
| `tabddpm` | Yes | [yandex-research/tab-ddpm](https://github.com/yandex-research/tab-ddpm) | Official implementation and widely reused. Reproducible, but upstream setup is dated and environment-sensitive. |
| `tabsyn` | Yes | [amazon-science/tabsyn](https://github.com/amazon-science/tabsyn) | Official implementation with a clear train/sample flow. Good research code, but dependency setup is somewhat brittle. |
| `tabdiff` | Yes | [MinkaiXu/TabDiff](https://github.com/minkaixu/tabdiff) | Official implementation with documented custom-dataset support. Good quality, but expects conda-style environment management and multiple evaluation environments. |
| `ctgan` | Yes | [sdv-dev/CTGAN](https://github.com/sdv-dev/CTGAN) | Best-maintained classic tabular generator in the set. Packaged, documented, tested, and actively released through the SDV ecosystem. |
| `tvae` | Yes | [sdv-dev/CTGAN](https://github.com/sdv-dev/CTGAN) | Shares the same maintained upstream as CTGAN. Easier to reproduce than most research-only baselines. |
| `smote` | Yes | [scikit-learn-contrib/imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) | Extremely reliable and actively maintained. Not a joint generative model, but a strong classical oversampling baseline. |
| `bn` | Yes, with caveats | [pgmpy/pgmpy](https://github.com/pgmpy/pgmpy) | High-quality, actively maintained library. Reproducibility is fine, but benchmark behavior depends strongly on exact modeling choices. |
| `nflow` | Yes, with caveats | [bayesiains/nflows](https://github.com/bayesiains/nflows) | Solid library, but more of a toolkit than a tabular benchmark package. Requires precise architecture and preprocessing choices for fair reproduction. |
| `arf` | Partial / yes via the Python wrapper path | [bips-hb/arf](https://github.com/bips-hb/arf) | Official implementation is R-first. In this repository, practical usability comes from the Python `arfpy` path instead of the original workflow. |
| `great` | Yes, but higher-maintenance | [tabularis-ai/be_great](https://github.com/tabularis-ai/be_great) | Maintained modern GReaT implementation. Better packaged than many paper repos, but still inherits HF/LLM runtime cost and sampling sensitivity. |
| `realtabformer` | Yes | [worldbank/REaLTabFormer](https://github.com/worldbank/REaLTabFormer) | Good engineering quality: tests, docs, packaging, and recent releases. Reproducible, but heavier than non-transformer baselines. |
| `nrgboost` | Yes, but relatively young | [Ajoo/nrgboost](https://github.com/Ajoo/nrgboost) | Official code and pip package exist. Runnable, but less battle-tested than CTGAN, TabDDPM, or imbalanced-learn style baselines. |
| `ctab-gan-plus` | Partial / usable research code | [Team-TUD/CTAB-GAN-Plus](https://github.com/Team-TUD/CTAB-GAN-Plus) | Official repo exists, but ergonomics are rougher than CTGAN/TVAE. Notebook-centric and still tied to legacy dependency assumptions. |
| `goggle` | Partial | No standalone upstream repo verified; this repo vendors the implementation via [amazon-science/tabsyn](https://github.com/amazon-science/tabsyn) | Runnable here, but operationally fragile because it depends on `dgl`, `torch-geometric`, and version-sensitive graph extensions. |
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
- `bn`
- `nflow`
- `nrgboost`

Real implementations, but less reproduction-friendly:

- `ctab-gan-plus`
- `goggle`
- `tabebm`
- `arf` if strict parity with the original R-first path is required

## Notes on Current Evidence

- The local adapter inventory and runtime notes are in:
  - [../standardized_tabular_diffusion/model_inventory.py](/Users/jpbach/Desktop/Standardized-Tabular-Diffusion/standardized_tabular_diffusion/model_inventory.py)
  - [runtime_status.md](/Users/jpbach/Desktop/Standardized-Tabular-Diffusion/docs/runtime_status.md)
- This assessment also checked current upstream repositories and package pages as of July 16, 2026.
- Example date-specific signals from upstream:
  - CTGAN latest release observed: February 13, 2026
  - REaLTabFormer latest release observed: January 4, 2026
  - pgmpy latest release observed: April 30, 2026
  - TabSyn README updates observed through June 20, 2024
  - TabDiff README updates observed through April 2025
