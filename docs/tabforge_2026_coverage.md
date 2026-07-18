# TabFORGE 2026 Coverage

This document tracks coverage for the benchmark list quoted from *Tabular Foundation Model for Generative Modelling* ([arXiv:2605.09424](https://arxiv.org/abs/2605.09424)).

Status labels:

- `Runnable adapter`: exposed through the shared `train / sample / evaluate` CLI surface.
- `Inventory only`: tracked in the repository inventory with source and reproduction notes, but not wired into the runnable generator registry.
- `Missing adapter`: source and reproduction notes are now tracked, but no runnable adapter exists yet.

## Coverage Table

| Model | Status | Notes |
| --- | --- | --- |
| `SMOTE` | Runnable adapter | Implemented as `smote`. |
| `TabSDS` | Runnable adapter | Implemented as `tabsds` via a local lightweight compatibility adapter. |
| `TVAE` | Runnable adapter | Implemented as `tvae`. |
| `GOGGLE` | Runnable adapter | Implemented as `goggle`. |
| `CTGAN` | Runnable adapter | Implemented as `ctgan`. |
| `NFlow` | Runnable adapter | Implemented as `nflow`. |
| `ARF` | Runnable adapter | Implemented as `arf`. |
| `TabDDPM` | Runnable adapter | Implemented as `tabddpm`. |
| `CDTD` | Missing adapter | Tracked as `cdtd`; official repo linked in the inventory. |
| `TabSyn` | Runnable adapter | Implemented as `tabsyn`. |
| `TabDiff` | Runnable adapter | Implemented as `tabdiff`. |
| `CTSyn` | Missing adapter | Tracked as `ctsyn`; no official repo verified in this pass. |
| `TabEBM` | Runnable adapter | Implemented as `tabebm`; sample generation remains gated by TabPFN access. |
| `NRGBoost` | Runnable adapter | Implemented as `nrgboost`. |
| `TabNAT` | Missing adapter | Tracked as `tabnat`; no official repo verified in this pass. |
| `TabularARGN` | Runnable adapter | Implemented as `tabularargn` via an optional-package adapter around `mostlyai-engine`. |
| `GReaT` | Runnable adapter | Implemented as `great`. |
| `Real-TabPFN-2.5` | Inventory only | Tracked as `realtabpfn`. |
| `TabDPT` | Inventory only | Tracked as `tabdpt`. |
| `Mitra` | Inventory only | Tracked as `mitra`. |
| `LimiX` | Inventory only | Tracked as `limix`. |
| `TabICL / TabICLv2` | Inventory only | Tracked as `tabicl` and `tabiclv2`. |

## CLI Support

To inspect the models tagged for this paper from the inventory:

```bash
python -m standardized_tabular_diffusion.cli list-model-inventory --benchmark tabforge-2026
```

To inspect a single paper entry:

```bash
python -m standardized_tabular_diffusion.cli show-model-inventory --model tabsds
```
