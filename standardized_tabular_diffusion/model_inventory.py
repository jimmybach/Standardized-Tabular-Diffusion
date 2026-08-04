from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ModelInventoryEntry:
    name: str
    family: str
    paradigm: str
    covered_by_papers: list[str]
    validation_level: str
    runnable_recommendation: str
    implementation_quality: str
    repository_url: str | None
    license_status: str = "not-reviewed"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        from standardized_tabular_diffusion.registry import get_adapter_spec

        try:
            adapter = get_adapter_spec(self.name)
        except KeyError:
            payload.update(
                {
                    "adapter_registered": False,
                    "source_authority": None,
                    "distribution_form": None,
                    "reproduction_target": None,
                    "modification_status": None,
                    "benchmark_track": "excluded",
                    "support_level": "unsupported",
                    "revision_status": "unresolved",
                    "evidence_records": [],
                }
            )
        else:
            adapter_payload = adapter.to_dict(self.name)
            payload.update(
                {
                    "adapter_registered": True,
                    "source_authority": adapter_payload["source_authority"],
                    "distribution_form": adapter_payload["distribution_form"],
                    "reproduction_target": adapter_payload["reproduction_target"],
                    "modification_status": adapter_payload["modification_status"],
                    "benchmark_track": adapter_payload["benchmark_track"],
                    "support_level": adapter_payload["support_level"],
                    "revision_status": adapter_payload["revision_status"],
                    "evidence_records": adapter_payload["evidence_records"],
                    "license_status": adapter_payload["license_status"],
                }
            )
        return payload


MODEL_INVENTORY: dict[str, ModelInventoryEntry] = {
    "tabddpm": ModelInventoryEntry(
        name="tabddpm",
        family="diffusion",
        paradigm="score / denoising diffusion",
        covered_by_papers=["tabstruct-2026", "tabula-2025", "tabforge-2026"],
        validation_level="native-parity-validated",
        runnable_recommendation="yes",
        implementation_quality="high",
        repository_url="https://github.com/yandex-research/tab-ddpm",
        notes=[
            "Already standardized in this repository.",
            "Official implementation is research-grade but reproducible and widely reused.",
            "Passed the retained Linux/Python 3.11 three-seed native-parity protocol; this is not an Official Results or release-support claim.",
        ],
    ),
    "tabsyn": ModelInventoryEntry(
        name="tabsyn",
        family="diffusion",
        paradigm="latent diffusion",
        covered_by_papers=["tabstruct-2026", "tabforge-2026"],
        validation_level="native-parity-validated",
        runnable_recommendation="yes",
        implementation_quality="high",
        repository_url="https://github.com/amazon-science/tabsyn",
        notes=[
            "Already standardized in this repository.",
            "Passed the retained Linux/Python 3.11 three-seed native-parity protocol; this is not an Official Results or release-support claim.",
            "Vendors several baseline implementations that can be reused for future adapters.",
        ],
    ),
    "tabdiff": ModelInventoryEntry(
        name="tabdiff",
        family="diffusion",
        paradigm="multimodal diffusion",
        covered_by_papers=["tabstruct-2026", "tabforge-2026"],
        validation_level="native-parity-validated",
        runnable_recommendation="yes",
        implementation_quality="high",
        repository_url="https://github.com/MinkaiXu/TabDiff",
        notes=[
            "Already standardized in this repository.",
            "Passed the retained Linux/Python 3.11 native-parity protocol; this is not an Official Results or release-support claim.",
        ],
    ),
    "great": ModelInventoryEntry(
        name="great",
        family="llm",
        paradigm="autoregressive transformer over row text",
        covered_by_papers=["tabstruct-2026", "tabula-2025", "tabforge-2026"],
        validation_level="adapter-complete",
        runnable_recommendation="yes",
        implementation_quality="high",
        repository_url="https://github.com/kathrinse/greaT",
        notes=[
            "Integrated through the vendored TabSyn baseline implementation and exposed through the standardized adapter registry.",
            "Still has a heavy Hugging Face dependency footprint compared with classical baselines.",
            "Runtime depends on access to a pretrained causal LM checkpoint such as distilgpt2.",
            "Training is reproducible through the shared CLI, and stronger ordered-column distilgpt2 runs now produce parseable samples.",
            "Tiny CPU smoke checkpoints still struggle to generate enough parseable rows for reliable sample validation.",
        ],
    ),
    "tabicl": ModelInventoryEntry(
        name="tabicl",
        family="foundation",
        paradigm="in-context tabular foundation model",
        covered_by_papers=[],
        validation_level="registered",
        runnable_recommendation="yes",
        implementation_quality="high",
        repository_url="https://github.com/soda-inria/tabicl",
        notes=[
            "Official open-source implementation exists and appears well maintained.",
            "Not yet wired into this repository.",
            "Upstream focuses on predictive in-context learning rather than direct tabular generation, so integrating it here would require a benchmark-specific generative adaptation path rather than a simple train/sample wrapper.",
        ],
    ),
    "tabiclv2": ModelInventoryEntry(
        name="tabiclv2",
        family="foundation",
        paradigm="scaled in-context tabular foundation model",
        covered_by_papers=["tabforge-2026"],
        validation_level="registered",
        runnable_recommendation="yes",
        implementation_quality="high",
        repository_url="https://github.com/soda-inria/tabicl",
        notes=[
            "Current flagship TabICL release is exposed through the official TabICL repository and documentation.",
            "Represents the stronger modern TabICL checkpoint family rather than the original ICML 2025 release alone.",
            "Still a predictive in-context learner rather than a native synthetic row generator, so it would need a benchmark-specific adaptation path here.",
        ],
    ),
    "tabpfn": ModelInventoryEntry(
        name="tabpfn",
        family="foundation",
        paradigm="prior-data fitted network",
        covered_by_papers=[],
        validation_level="registered",
        runnable_recommendation="yes",
        implementation_quality="high",
        repository_url="https://github.com/PriorLabs/TabPFN",
        notes=[
            "Official open-source TabPFN repository is actively maintained and exposes current model versions through a Python package.",
            "Important baseline in the tabular foundation-model landscape and the parent lineage behind later real-data continued-pretraining variants.",
            "Primary upstream use case is zero-shot prediction rather than synthetic row generation, so it does not slot directly into this repository's generative adapter contract.",
        ],
    ),
    "realtabpfn": ModelInventoryEntry(
        name="realtabpfn",
        family="foundation",
        paradigm="continued-pretrained prior-data fitted network",
        covered_by_papers=["tabforge-2026"],
        validation_level="registered",
        runnable_recommendation="partial",
        implementation_quality="high",
        repository_url="https://github.com/PriorLabs/TabPFN",
        notes=[
            "Official TabPFN repository exposes real-data-finetuned TabPFN-2.5 checkpoints alongside the synthetic-only variants.",
            "Strong predictive foundation-model baseline, but not a native tabular generator in the same train/sample sense as the other adapters here.",
            "Practical usage may require gated model access, authentication, and acceptance of non-commercial terms for some real-data checkpoints.",
        ],
    ),
    "tabdpt": ModelInventoryEntry(
        name="tabdpt",
        family="foundation",
        paradigm="in-context tabular foundation model trained on real data",
        covered_by_papers=["tabforge-2026"],
        validation_level="registered",
        runnable_recommendation="yes",
        implementation_quality="high",
        repository_url="https://github.com/layer6ai-labs/TabDPT",
        notes=[
            "Official open-source implementation exists, with a maintained PyPI package and Hugging Face model card.",
            "Designed for zero-shot classification and regression on unseen tables rather than direct synthetic row generation.",
            "Likely feasible to benchmark as a predictive foundation-model reference, but not a drop-in fit for this repository's generative train/sample/evaluate contract.",
        ],
    ),
    "tabfm": ModelInventoryEntry(
        name="tabfm",
        family="foundation",
        paradigm="hybrid row-column attention plus in-context tabular foundation model",
        covered_by_papers=[],
        validation_level="registered",
        runnable_recommendation="yes",
        implementation_quality="high",
        repository_url="https://github.com/google-research/tabfm",
        notes=[
            "Official open-source repository from Google Research is available.",
            "Supports zero-shot classification and regression with scikit-learn-compatible wrappers and downloadable pretrained weights.",
            "Method is a predictive foundation model rather than a native synthetic tabular generator, so it does not map directly onto this repository's generative train/sample contract.",
        ],
    ),
    "mothernet": ModelInventoryEntry(
        name="mothernet",
        family="foundation",
        paradigm="hypernetwork tabular foundation model",
        covered_by_papers=[],
        validation_level="registered",
        runnable_recommendation="partial",
        implementation_quality="medium",
        repository_url="https://github.com/microsoft/ticl",
        notes=[
            "Official implementation is available through Microsoft's Tabular In-Context Learning repository.",
            "MotherNet is presented there as a foundational hypernetwork for tabular classification with sklearn-style prediction wrappers.",
            "Relevant as a predictive foundation-model reference, but not a native tabular generator for this repository's current train/sample benchmark contract.",
        ],
    ),
    "gamformer": ModelInventoryEntry(
        name="gamformer",
        family="foundation",
        paradigm="interpretable in-context tabular foundation model",
        covered_by_papers=[],
        validation_level="registered",
        runnable_recommendation="partial",
        implementation_quality="medium",
        repository_url="https://github.com/microsoft/ticl",
        notes=[
            "Official implementation is listed in Microsoft's Tabular In-Context Learning repository.",
            "Positioned as an interpretable additive-model-style in-context learner rather than a generative synthesizer.",
            "Useful as a foundation-model reference, but outside this repository's direct generative adapter shape.",
        ],
    ),
    "causalfm": ModelInventoryEntry(
        name="causalfm",
        family="foundation",
        paradigm="prior-data fitted network for causal inference",
        covered_by_papers=[],
        validation_level="registered",
        runnable_recommendation="partial",
        implementation_quality="medium",
        repository_url="https://github.com/yccm/CausalFM",
        notes=[
            "Official implementation is available for CausalFM and its associated toolkit.",
            "The method is a PFN-style foundation model specialized for causal inference settings such as CATE, IV, and front-door adjustment.",
            "Relevant as a foundation-model reference, but it is outside this repository's synthetic tabular generation contract and benchmark focus.",
        ],
    ),
    "tabflex": ModelInventoryEntry(
        name="tabflex",
        family="foundation",
        paradigm="linear-attention extension of tabular in-context learning",
        covered_by_papers=[],
        validation_level="registered",
        runnable_recommendation="partial",
        implementation_quality="medium",
        repository_url="https://github.com/microsoft/ticl",
        notes=[
            "Official implementation is listed in Microsoft's Tabular In-Context Learning repository.",
            "Described there as a TabPFN extension using linear attention to scale to more features, models, and classes.",
            "Important predictive foundation-model reference, but not a synthetic-row generator in the sense expected by this repository.",
        ],
    ),
    "tabula": ModelInventoryEntry(
        name="tabula",
        family="llm",
        paradigm="language-model-based tabular synthesis",
        covered_by_papers=["tabula-2025"],
        validation_level="adapter-complete",
        runnable_recommendation="partial",
        implementation_quality="medium",
        repository_url="https://github.com/zhao-zilong/Tabula",
        notes=[
            "Official repository exists for the generative model TabuLa: Harnessing Language Models for Tabular Data Synthesis.",
            "Method is directly relevant to this repository's generative benchmark scope.",
            "This repository now exposes a local standardized TabuLa-compatible adapter built on top of Transformers rather than depending on the upstream notebook flow directly.",
            "The standardized path is practical but should be treated as a compatibility implementation inspired by the paper's training recipe rather than a bit-for-bit wrapper of the original repository.",
        ],
    ),
    "transtab": ModelInventoryEntry(
        name="transtab",
        family="foundation",
        paradigm="transferable tabular transformer across variable-column tables",
        covered_by_papers=[],
        validation_level="registered",
        runnable_recommendation="yes",
        implementation_quality="high",
        repository_url="https://github.com/RyanWangZf/transtab",
        notes=[
            "Official repository and PyPI package exist for TransTab.",
            "Relevant as a cross-table pretraining and transfer baseline for tabular foundation-model discussions.",
            "Primary upstream use cases are prediction, transfer learning, and representation learning across tables rather than synthetic row generation.",
        ],
    ),
    "smote": ModelInventoryEntry(
        name="smote",
        family="traditional",
        paradigm="interpolation / oversampling",
        covered_by_papers=["tabstruct-2026", "tabforge-2026"],
        validation_level="adapter-complete",
        runnable_recommendation="yes",
        implementation_quality="high",
        repository_url="https://github.com/scikit-learn-contrib/imbalanced-learn",
        notes=[
            "Mature scikit-learn style implementation with strong maintenance.",
            "Not a true joint generative model; better treated as a classical oversampling baseline.",
            "Best integrated as a lightweight adapter with explicit caveats in benchmark tables.",
        ],
    ),
    "stasy": ModelInventoryEntry(
        name="stasy",
        family="diffusion",
        paradigm="score-based SDE diffusion",
        covered_by_papers=[],
        validation_level="adapter-complete",
        runnable_recommendation="partial",
        implementation_quality="medium",
        repository_url="https://github.com/amazon-science/tabsyn",
        notes=[
            "Integrated through the vendored TabSyn baseline implementation.",
            "Uses the upstream STaSy training and sampling flow via TabSyn's shared baseline dispatcher.",
            "Wired into the standardized train/sample/evaluate contract, but not yet smoke-validated in this repository.",
        ],
    ),
    "bn": ModelInventoryEntry(
        name="bn",
        family="traditional",
        paradigm="bayesian network",
        covered_by_papers=["tabstruct-2026"],
        validation_level="adapter-complete",
        runnable_recommendation="yes",
        implementation_quality="medium",
        repository_url="https://github.com/pgmpy/pgmpy",
        notes=[
            "Integrated as a direct pgmpy-backed Bayesian-network adapter rather than through Synthcity.",
            "A local adult end-to-end run was previously reported, but formal Linux/Python 3.11 smoke evidence is not yet recorded.",
            "Sampling can drop low-variance variables in the learned graph, so the adapter restores missing columns with fitted fallback states.",
        ],
    ),
    "codi": ModelInventoryEntry(
        name="codi",
        family="diffusion",
        paradigm="co-evolving continuous/discrete diffusion",
        covered_by_papers=[],
        validation_level="adapter-complete",
        runnable_recommendation="partial",
        implementation_quality="medium",
        repository_url="https://github.com/amazon-science/tabsyn",
        notes=[
            "Integrated through the vendored TabSyn baseline implementation.",
            "Uses TabSyn's shared baseline dispatcher rather than a standalone upstream package.",
            "Wired into the standardized train/sample/evaluate contract, but not yet smoke-validated in this repository.",
        ],
    ),
    "ctab-gan": ModelInventoryEntry(
        name="ctab-gan",
        family="gan",
        paradigm="conditional tabular GAN with tailored preprocessing",
        covered_by_papers=[],
        validation_level="adapter-complete",
        runnable_recommendation="partial",
        implementation_quality="medium",
        repository_url="https://github.com/Team-TUD/CTAB-GAN",
        notes=[
            "Integrated against the vendored CTAB-GAN implementation already present in this repository.",
            "Legacy research-code ergonomics and dependency assumptions are similar to CTAB-GAN+.",
            "Wired into the standardized train/sample/evaluate contract, but not yet smoke-validated in this repository.",
        ],
    ),
    "tvae": ModelInventoryEntry(
        name="tvae",
        family="vae",
        paradigm="variational autoencoder",
        covered_by_papers=["tabstruct-2026", "tabforge-2026"],
        validation_level="adapter-complete",
        runnable_recommendation="yes",
        implementation_quality="high",
        repository_url="https://github.com/sdv-dev/CTGAN",
        notes=[
            "The adapter now targets TVAE from the checksum-pinned official ctgan 0.12.1 wheel.",
            "The locally modified 0.5.2.dev0 snapshot and its obsolete wrappers were removed rather than presented as the official implementation.",
            "A mandatory Linux/Python 3.11 native-parity protocol is pending before status promotion.",
            "Version 0.12.1 uses BUSL-1.1; Official Results and release support require a separate license decision.",
        ],
    ),
    "goggle": ModelInventoryEntry(
        name="goggle",
        family="graph",
        paradigm="graph generative model",
        covered_by_papers=["tabstruct-2026", "tabforge-2026"],
        validation_level="adapter-complete",
        runnable_recommendation="yes",
        implementation_quality="medium",
        repository_url="https://github.com/amazon-science/tabsyn",
        notes=[
            "Integrated against the vendored TabSyn baseline implementation with local compatibility fixes for modern sklearn and Python packaging.",
            "A local adult end-to-end run was previously reported, but formal Linux/Python 3.11 smoke evidence is not yet recorded.",
            "Still the most fragile of the implemented adapters because it depends on DGL, torch-geometric, and version-sensitive graph extensions.",
        ],
    ),
    "ctgan": ModelInventoryEntry(
        name="ctgan",
        family="gan",
        paradigm="conditional GAN",
        covered_by_papers=["tabstruct-2026", "tabula-2025", "tabforge-2026"],
        validation_level="native-parity-validated",
        runnable_recommendation="yes",
        implementation_quality="high",
        repository_url="https://github.com/sdv-dev/CTGAN",
        notes=[
            "The adapter now targets the checksum-pinned official ctgan 0.12.1 wheel instead of the legacy embedded 0.5.2.dev0 source snapshot.",
            "The mandatory Linux/Python 3.11 native-parity protocol passed all exact comparisons for seeds 0, 19, and 73 in GitHub Actions run 30910275922.",
            "Version 0.12.1 uses BUSL-1.1; Official Results and release support require a separate license decision.",
        ],
    ),
    "nflow": ModelInventoryEntry(
        name="nflow",
        family="flow",
        paradigm="normalizing flow",
        covered_by_papers=["tabstruct-2026", "tabforge-2026"],
        validation_level="adapter-complete",
        runnable_recommendation="yes",
        implementation_quality="medium",
        repository_url="https://github.com/bayesiains/nflows",
        notes=[
            "Integrated as a direct nflows-backed masked autoregressive flow adapter rather than through Synthcity.",
            "A local adult end-to-end run was previously reported, but formal Linux/Python 3.11 smoke evidence is not yet recorded.",
            "The benchmark label is generic, so the registry should continue documenting the exact flow architecture used in standardized runs.",
        ],
    ),
    "arf": ModelInventoryEntry(
        name="arf",
        family="tree",
        paradigm="adversarial random forest",
        covered_by_papers=["tabstruct-2026", "tabforge-2026"],
        validation_level="adapter-complete",
        runnable_recommendation="yes",
        implementation_quality="high",
        repository_url="https://github.com/bips-hb/arf",
        notes=[
            "Integrated through the Python arfpy package rather than the original R-first workflow.",
            "The standardized adapter uses ARF density estimation plus FORGE sampling and fits the shared train/sample/evaluate contract cleanly.",
            "A strong non-neural baseline with much lower operational overhead than the original R path.",
        ],
    ),
    "tabebm": ModelInventoryEntry(
        name="tabebm",
        family="energy-based",
        paradigm="class-conditional energy-based model",
        covered_by_papers=["tabstruct-2026", "tabforge-2026"],
        validation_level="adapter-complete",
        runnable_recommendation="partial",
        implementation_quality="medium",
        repository_url="https://github.com/andreimargeloiu/TabEBM",
        notes=[
            "Integrated as a classification-only adapter with class-conditional sampling semantics.",
            "Runtime still depends on a working TabPFN setup and accepted gated-model terms from Prior Labs on Hugging Face.",
            "More operationally fragile than the other implemented baselines because it sits on top of a gated foundation-model dependency.",
        ],
    ),
    "nrgboost": ModelInventoryEntry(
        name="nrgboost",
        family="energy-based",
        paradigm="energy-based boosted trees",
        covered_by_papers=["tabstruct-2026", "tabforge-2026"],
        validation_level="adapter-complete",
        runnable_recommendation="yes",
        implementation_quality="medium",
        repository_url="https://github.com/Ajoo/nrgboost",
        notes=[
            "Official code and pip package are available.",
            "Requires Python 3.10+ and currently supports Linux/macOS, with source builds needing OpenMP when wheels are unavailable.",
            "The standardized adapter expects the optional `nrgboost` package to be installed locally.",
            "Promising integration target if we want broader methodological coverage beyond neural generators.",
        ],
    ),
    "ctab-gan-plus": ModelInventoryEntry(
        name="ctab-gan-plus",
        family="gan",
        paradigm="conditional tabular GAN with tailored preprocessing",
        covered_by_papers=["tabula-2025"],
        validation_level="adapter-complete",
        runnable_recommendation="yes",
        implementation_quality="medium",
        repository_url="https://github.com/Team-TUD/CTAB-GAN-Plus",
        notes=[
            "Official code exists and is commonly reused in benchmarks.",
            "Research-code ergonomics are rougher than SDV CTGAN/TVAE and preprocessing assumptions are more bespoke.",
            "The standardized adapter uses the vendored implementation in this repository and still depends on legacy extras such as `dython`.",
            "Still worth integrating because it is a recurring tabular benchmark baseline.",
        ],
    ),
    "realtabformer": ModelInventoryEntry(
        name="realtabformer",
        family="llm",
        paradigm="autoregressive / seq2seq transformer",
        covered_by_papers=["tabula-2025"],
        validation_level="adapter-complete",
        runnable_recommendation="yes",
        implementation_quality="high",
        repository_url="https://github.com/worldbank/REaLTabFormer",
        notes=[
            "Well-packaged repository with PyPI support, docs, tests, and both tabular and relational modes.",
            "Heavier than classical baselines because of transformer training cost and tokenizer/model setup.",
            "The standardized adapter expects the optional `realtabformer` package to be installed locally.",
            "Very good candidate for the next LLM-style adapter after GReaT.",
        ],
    ),
    "tabsds": ModelInventoryEntry(
        name="tabsds",
        family="traditional",
        paradigm="non-parametric rank-and-shuffle tabular synthesis",
        covered_by_papers=["tabforge-2026"],
        validation_level="adapter-complete",
        runnable_recommendation="partial",
        implementation_quality="medium",
        repository_url="https://github.com/echaibub/TabSDS",
        notes=[
            "Official R and Python implementations are linked from the ICML 2025 paper and OpenReview page.",
            "This repository now exposes a local lightweight compatibility adapter inspired by the TabSDS method.",
            "The current adapter is practical and runnable, but it should be treated as an approximation rather than a bit-for-bit wrapper around the official upstream implementation.",
        ],
    ),
    "cdtd": ModelInventoryEntry(
        name="cdtd",
        family="diffusion",
        paradigm="continuous diffusion for mixed-type tabular data",
        covered_by_papers=["tabforge-2026"],
        validation_level="registered",
        runnable_recommendation="yes",
        implementation_quality="high",
        repository_url="https://github.com/muellermarkus/cdtd",
        notes=[
            "Official implementation exists and is explicitly referenced by the paper.",
            "The current repository only references CDTD in comments inside vendored preprocessing code; no adapter is wired yet.",
            "A real integration would likely require vendoring or directly depending on the upstream project.",
        ],
    ),
    "ctsyn": ModelInventoryEntry(
        name="ctsyn",
        family="diffusion",
        paradigm="cross-table latent diffusion foundation model",
        covered_by_papers=["tabforge-2026"],
        validation_level="registered",
        runnable_recommendation="partial",
        implementation_quality="unknown",
        repository_url=None,
        notes=[
            "The paper is publicly visible, but no official source repository was verified in this pass.",
            "Because CTSyn is a cross-table generative foundation model, integration would also require decisions about how to represent multi-table datasets in the current single-table benchmark interface.",
        ],
    ),
    "tabnat": ModelInventoryEntry(
        name="tabnat",
        family="autoregressive",
        paradigm="tabular-specific autoregressive generator",
        covered_by_papers=["tabforge-2026"],
        validation_level="registered",
        runnable_recommendation="partial",
        implementation_quality="unknown",
        repository_url=None,
        notes=[
            "Included in the paper's benchmark list, but no official repository was verified in this pass.",
            "Without a verified upstream implementation surface, this remains a tracked paper gap rather than a runnable adapter candidate.",
        ],
    ),
    "tabularargn": ModelInventoryEntry(
        name="tabularargn",
        family="autoregressive",
        paradigm="flexible autoregressive tabular synthesizer",
        covered_by_papers=["tabforge-2026"],
        validation_level="adapter-complete",
        runnable_recommendation="partial",
        implementation_quality="high",
        repository_url="https://github.com/mostly-ai/mostlyai-engine",
        notes=[
            "A strong maintained implementation exists via MOSTLY AI's open-source engine, with direct `fit()` and `sample()` support.",
            "This repository now exposes a standardized optional-package adapter around that engine.",
            "The adapter depends on installing `mostlyai-engine` locally and has not yet been smoke-validated in this repository.",
        ],
    ),
    "mitra": ModelInventoryEntry(
        name="mitra",
        family="foundation",
        paradigm="mixed-synthetic-priors tabular foundation model",
        covered_by_papers=["tabforge-2026"],
        validation_level="registered",
        runnable_recommendation="yes",
        implementation_quality="high",
        repository_url="https://github.com/autogluon/autogluon",
        notes=[
            "Mitra is available through AutoGluon and documented in the official AutoGluon foundation-model tutorials.",
            "It is a predictive tabular foundation model rather than a native synthetic-row generator, so it does not directly fit the current generator adapter contract.",
        ],
    ),
    "limix": ModelInventoryEntry(
        name="limix",
        family="foundation",
        paradigm="generalist structured-data foundation model",
        covered_by_papers=["tabforge-2026"],
        validation_level="registered",
        runnable_recommendation="partial",
        implementation_quality="medium",
        repository_url="https://github.com/limix-ldm-ai/LimiX",
        notes=[
            "An official repository exists for LimiX.",
            "The model is oriented toward broad structured-data prediction and inference tasks rather than native synthetic table generation.",
            "Relevant to the paper's PFN-style foundation-model comparison set, but outside this repository's current runnable generator surface.",
        ],
    ),
}


def get_inventory_entry(model_name: str) -> ModelInventoryEntry:
    return MODEL_INVENTORY[model_name]


def list_inventory(
    *,
    benchmark: str | None = None,
    family: str | None = None,
    validation_level: str | None = None,
) -> list[ModelInventoryEntry]:
    entries = sorted(MODEL_INVENTORY.values(), key=lambda entry: entry.name)
    if benchmark is not None:
        entries = [entry for entry in entries if benchmark in entry.covered_by_papers]
    if family is not None:
        entries = [entry for entry in entries if entry.family == family]
    if validation_level is not None:
        entries = [entry for entry in entries if entry.validation_level == validation_level]
    return entries


def list_inventory_names(
    *,
    benchmark: str | None = None,
    family: str | None = None,
    validation_level: str | None = None,
) -> list[str]:
    return [
        entry.name
        for entry in list_inventory(
            benchmark=benchmark,
            family=family,
            validation_level=validation_level,
        )
    ]
