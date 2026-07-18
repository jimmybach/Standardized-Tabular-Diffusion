# Smoke Presets

These presets are small, reproducible configs for quickly validating newly integrated models without running the full benchmark protocol.

## Included Presets

- [configs/smoke/nrgboost-adult-smoke.json](/Users/jpbach/Desktop/Standardized-Tabular-Diffusion/configs/smoke/nrgboost-adult-smoke.json)
- [configs/smoke/ctab-gan-plus-adult-smoke.json](/Users/jpbach/Desktop/Standardized-Tabular-Diffusion/configs/smoke/ctab-gan-plus-adult-smoke.json)
- [configs/smoke/realtabformer-adult-tiny.json](/Users/jpbach/Desktop/Standardized-Tabular-Diffusion/configs/smoke/realtabformer-adult-tiny.json)
- [configs/smoke/bn-adult-smoke.json](/Users/jpbach/Desktop/Standardized-Tabular-Diffusion/configs/smoke/bn-adult-smoke.json)
- [configs/smoke/nflow-adult-smoke.json](/Users/jpbach/Desktop/Standardized-Tabular-Diffusion/configs/smoke/nflow-adult-smoke.json)
- [configs/smoke/goggle-adult-smoke.json](/Users/jpbach/Desktop/Standardized-Tabular-Diffusion/configs/smoke/goggle-adult-smoke.json)
- [configs/smoke/arf-adult-smoke.json](/Users/jpbach/Desktop/Standardized-Tabular-Diffusion/configs/smoke/arf-adult-smoke.json)
- [configs/smoke/arf-shoppers-smoke.json](/Users/jpbach/Desktop/Standardized-Tabular-Diffusion/configs/smoke/arf-shoppers-smoke.json)
- [configs/smoke/great-adult-train-smoke.json](/Users/jpbach/Desktop/Standardized-Tabular-Diffusion/configs/smoke/great-adult-train-smoke.json)
- [configs/smoke/great-adult-tiny.json](/Users/jpbach/Desktop/Standardized-Tabular-Diffusion/configs/smoke/great-adult-tiny.json)
- [configs/smoke/great-adult-distilgpt2-tiny.json](/Users/jpbach/Desktop/Standardized-Tabular-Diffusion/configs/smoke/great-adult-distilgpt2-tiny.json)
- [configs/smoke/great-adult-distilgpt2-strong.json](/Users/jpbach/Desktop/Standardized-Tabular-Diffusion/configs/smoke/great-adult-distilgpt2-strong.json)
- [configs/smoke/tabebm-adult-smoke.json](/Users/jpbach/Desktop/Standardized-Tabular-Diffusion/configs/smoke/tabebm-adult-smoke.json)
- [configs/smoke/tabebm-adult-gated-sample.json](/Users/jpbach/Desktop/Standardized-Tabular-Diffusion/configs/smoke/tabebm-adult-gated-sample.json)

At the moment there is no dedicated smoke preset checked in for `tabula`, `ctab-gan`, `stasy`, or `codi`, even though those adapters are now integrated into the shared registry.

## Usage

Run a preset with:

```bash
python -m standardized_tabular_diffusion.cli run --config configs/smoke/nrgboost-adult-smoke.json
```

or:

```bash
python -m standardized_tabular_diffusion.cli run --config configs/smoke/ctab-gan-plus-adult-smoke.json
```

For `realtabformer`, use the tiny preset first:

```bash
python -m standardized_tabular_diffusion.cli run --config configs/smoke/realtabformer-adult-tiny.json
```

For the newly wired structured baselines:

```bash
python -m standardized_tabular_diffusion.cli run --config configs/smoke/bn-adult-smoke.json
python -m standardized_tabular_diffusion.cli run --config configs/smoke/nflow-adult-smoke.json
python -m standardized_tabular_diffusion.cli run --config configs/smoke/goggle-adult-smoke.json
python -m standardized_tabular_diffusion.cli run --config configs/smoke/arf-adult-smoke.json
python -m standardized_tabular_diffusion.cli run --config configs/smoke/arf-shoppers-smoke.json
python -m standardized_tabular_diffusion.cli run --config configs/smoke/great-adult-train-smoke.json
python -m standardized_tabular_diffusion.cli run --config configs/smoke/great-adult-tiny.json
python -m standardized_tabular_diffusion.cli run --config configs/smoke/great-adult-distilgpt2-tiny.json
python -m standardized_tabular_diffusion.cli run --config configs/smoke/great-adult-distilgpt2-strong.json
```

For `tabebm`, the preset also needs Prior Labs TabPFN access:

```bash
python -m standardized_tabular_diffusion.cli run --config configs/smoke/tabebm-adult-smoke.json
python -m standardized_tabular_diffusion.cli run --config configs/smoke/tabebm-adult-gated-sample.json
```

## Notes

- `nrgboost-adult-smoke.json` is a real end-to-end validated smoke preset.
- `ctab-gan-plus-adult-smoke.json` is a real end-to-end validated smoke preset.
- `bn-adult-smoke.json`, `nflow-adult-smoke.json`, and `goggle-adult-smoke.json` are real end-to-end validated smoke presets for the newly added baseline families.
- `arf-adult-smoke.json` is intended to be a fast end-to-end validation preset for the ARF adapter.
- `arf-shoppers-smoke.json` is a second-dataset ARF validation preset to confirm the adapter is not adult-specific.
- `great-adult-train-smoke.json` is the reliable GReaT integration check when you mainly want to validate training and artifact creation.
- `great-adult-tiny.json` uses `sshleifer/tiny-gpt2` so the first sample-oriented adapter validation run is practical on CPU, and now includes fallback temperature and `max_length` schedules.
- `great-adult-distilgpt2-tiny.json` is the stronger-base follow-up check when you want to distinguish checkpoint weakness from adapter/runtime issues.
- `great-adult-distilgpt2-strong.json` is the first GReaT preset intended to be sample-capable rather than train-only; it relies on ordered-column training and first-column prompting.
- `tabebm-adult-smoke.json` is now the train-only integration check for TabEBM.
- `tabebm-adult-gated-sample.json` is the explicit opt-in sample path for machines with accepted TabPFN gated-model access.
- `realtabformer-adult-tiny.json` is intentionally more conservative than the other presets:
  it disables sensitivity analysis, disables external reporting, and samples a small training subset with `max_train_rows`.
- The `realtabformer` tiny preset is meant for local integration checks, not meaningful benchmark numbers.
- `tabula` is integrated in code, but it does not yet have a committed smoke preset; use `example-config --model tabula` as the starting point for local validation runs.
