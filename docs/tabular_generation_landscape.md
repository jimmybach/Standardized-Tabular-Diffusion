# Tabular Generation Landscape

This document captures the current integration roadmap for the benchmarking repository and a higher-level map of how the tabular generation literature has evolved.

## Benchmark Scope

### Models already standardized here

- `tabddpm`
- `tabsyn`
- `tabdiff`
- `stasy`
- `codi`
- `great`
- `smote`
- `bn`
- `tvae`
- `goggle`
- `ctgan`
- `ctab-gan`
- `nflow`
- `arf`
- `tabebm`
- `nrgboost`
- `ctab-gan-plus`
- `realtabformer`

### Remaining caveat

- `tabebm` is standardized in code, but its runtime still depends on authenticated access to Prior Labs' gated TabPFN model.

## Recommended Integration Waves

### Wave 1: High-confidence, Python-native, benchmark-stable

- `smote`
- `ctgan`
- `tvae`
- `ctab-gan-plus`
- `realtabformer`
- `nrgboost`
- `arf`

These all have a credible path to a reproducible wrapper with a consistent `train / sample / evaluate` contract.

### Wave 2: Runnable, but with heavier dependency or design ambiguity

- `bn`
- `goggle`
- `nflow`
- `great`
- `ctab-gan`
- `stasy`
- `codi`

These are practical to standardize, but they carry more environment complexity than the first wave.

### Wave 3: Scientifically valuable, but operationally awkward

- `tabebm`

`tabebm` is scientifically relevant, but its runtime remains awkward because it depends on a gated TabPFN foundation model and class-conditional generation semantics.

## Taxonomy

### Traditional / statistical / resampling

- `smote`
- `bn`

These methods predate the recent deep-generation wave. They remain useful because they are cheap, easy to run, and often surprisingly competitive on utility-oriented metrics.

### VAE / GAN

- `tvae`
- `ctgan`
- `ctab-gan`
- `ctab-gan-plus`

This family defined the first widely adopted deep-learning baselines for tabular generation. Most later papers still compare against them because they established the default benchmark stack.

### Flow-based

- `nflow`

Normalizing flows matter because they keep explicit likelihood structure, which makes them conceptually different from adversarial and diffusion models even when benchmark performance is similar.

### Tree-based / classical non-neural generators

- `arf`

This line is important because tabular learning has always had a strong tree-model tradition, and generative tree methods are one of the few credible alternatives to neural generators.

### Diffusion

- `tabddpm`
- `tabsyn`
- `tabdiff`
- `stasy`
- `codi`

Diffusion models became the dominant modern research direction once the field moved beyond GAN-first baselines, largely because they improved robustness on heterogeneous mixed-type tables.

### LLM-based / transformer autoregressive

- `great`
- `realtabformer`

This branch emerged when researchers started treating rows as text sequences. It brought flexible conditioning and support for high-cardinality categories, at the cost of much heavier training and tokenization complexity.

### Graph-based

- `goggle`

This line reflects a shift toward explicitly modeling inter-column relations rather than only learning them implicitly in latent space.

### Energy-based

- `tabebm`
- `nrgboost`

Energy-based models are the clearest sign that the literature is expanding beyond the GAN/VAE/diffusion triad. They matter because they push toward explicit density structure, flexible inference, and closer ties to classical tabular inductive biases.

## Literature Evolution

### Phase 1: Classical synthetic data and probabilistic structure

Early tabular synthesis centered on statistical generators, resampling, copulas, and Bayesian-network-style models. The emphasis was practical data sharing and imbalanced-learning support rather than expressive end-to-end representation learning.

### Phase 2: Deep tabular generation through GANs and VAEs

The first major deep-learning wave adapted image-style generative modeling to mixed-type tables. `CTGAN`, `TVAE`, `CTAB-GAN`, and `CTAB-GAN+` belong here. This phase established the preprocessing-heavy, column-type-aware benchmark recipe that many papers still inherit.

### Phase 3: Diffusion becomes the new default

`TabDDPM`, then later `TabSyn`, `STaSy`, `CoDi`, and `TabDiff`, marked the shift toward diffusion as the strongest general-purpose paradigm. This phase focused on mixed continuous/categorical generation, better mode coverage, and more stable training than GAN-based baselines.

### Phase 4: Autoregressive language-model framing

`GReaT`, `REaLTabFormer`, and `TabuLa` reframed tables as text-like sequences. This brought a new axis of flexibility: few-shot reuse, prompt-like conditioning, and easier handling of high-cardinality categorical values. It also introduced new costs: tokenizer decisions, sequence length growth, and much slower training.

### Phase 5: Structure-aware and non-neural diversification

Recent work such as `GOGGLE`, `TabEBM`, `NRGBoost`, and the broader `TabStruct` benchmark reflects a maturing field. The focus is no longer just whether a model matches utility metrics, but whether it respects inter-feature structure, supports reliable inference, and offers a realistic trade-off between fidelity, privacy, and reproducibility.

## Practical Takeaways

- The field has moved from "can we generate plausible rows?" to "what inductive bias best preserves tabular structure?"
- Diffusion is the current strongest general-purpose research family.
- GAN and VAE baselines remain mandatory because the literature still anchors comparisons to them.
- LLM-based methods are important strategically, but they are not yet the easiest models to standardize or reproduce.
- Energy-based and tree-based generators are increasingly important for future research because they may fit tabular structure better than image-derived neural recipes.
