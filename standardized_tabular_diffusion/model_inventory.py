from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ModelInventoryEntry:
    name: str
    family: str
    paradigm: str
    covered_by_papers: list[str]
    standardized_status: str
    runnable_recommendation: str
    implementation_quality: str
    repository_url: str | None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


MODEL_INVENTORY: dict[str, ModelInventoryEntry] = {
    "tabddpm": ModelInventoryEntry(
        name="tabddpm",
        family="diffusion",
        paradigm="score / denoising diffusion",
        covered_by_papers=["tabstruct-2026", "tabula-2025"],
        standardized_status="implemented",
        runnable_recommendation="yes",
        implementation_quality="high",
        repository_url="https://github.com/yandex-research/tab-ddpm",
        notes=[
            "Already standardized in this repository.",
            "Official implementation is research-grade but reproducible and widely reused.",
        ],
    ),
    "tabsyn": ModelInventoryEntry(
        name="tabsyn",
        family="diffusion",
        paradigm="latent diffusion",
        covered_by_papers=["tabstruct-2026"],
        standardized_status="implemented",
        runnable_recommendation="yes",
        implementation_quality="high",
        repository_url="https://github.com/amazon-science/tabsyn",
        notes=[
            "Already standardized in this repository.",
            "Vendors several baseline implementations that can be reused for future adapters.",
        ],
    ),
    "tabdiff": ModelInventoryEntry(
        name="tabdiff",
        family="diffusion",
        paradigm="multimodal diffusion",
        covered_by_papers=["tabstruct-2026"],
        standardized_status="implemented",
        runnable_recommendation="yes",
        implementation_quality="high",
        repository_url="https://github.com/MinkaiXu/TabDiff",
        notes=[
            "Already standardized in this repository.",
        ],
    ),
    "great": ModelInventoryEntry(
        name="great",
        family="llm",
        paradigm="autoregressive transformer over row text",
        covered_by_papers=["tabstruct-2026", "tabula-2025"],
        standardized_status="implemented",
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
    "smote": ModelInventoryEntry(
        name="smote",
        family="traditional",
        paradigm="interpolation / oversampling",
        covered_by_papers=["tabstruct-2026"],
        standardized_status="implemented",
        runnable_recommendation="yes",
        implementation_quality="high",
        repository_url="https://github.com/scikit-learn-contrib/imbalanced-learn",
        notes=[
            "Mature scikit-learn style implementation with strong maintenance.",
            "Not a true joint generative model; better treated as a classical oversampling baseline.",
            "Best integrated as a lightweight adapter with explicit caveats in benchmark tables.",
        ],
    ),
    "bn": ModelInventoryEntry(
        name="bn",
        family="traditional",
        paradigm="bayesian network",
        covered_by_papers=["tabstruct-2026"],
        standardized_status="implemented",
        runnable_recommendation="yes",
        implementation_quality="medium",
        repository_url="https://github.com/pgmpy/pgmpy",
        notes=[
            "Integrated as a direct pgmpy-backed Bayesian-network adapter rather than through Synthcity.",
            "End-to-end adult smoke validation now passes through the shared CLI.",
            "Sampling can drop low-variance variables in the learned graph, so the adapter restores missing columns with fitted fallback states.",
        ],
    ),
    "tvae": ModelInventoryEntry(
        name="tvae",
        family="vae",
        paradigm="variational autoencoder",
        covered_by_papers=["tabstruct-2026"],
        standardized_status="implemented",
        runnable_recommendation="yes",
        implementation_quality="high",
        repository_url="https://github.com/sdv-dev/CTGAN",
        notes=[
            "Official maintained implementation ships in the CTGAN project and is also exposed through SDV.",
            "Very reasonable next integration target because it already has a pip-installable API and consistent preprocessing expectations.",
        ],
    ),
    "goggle": ModelInventoryEntry(
        name="goggle",
        family="graph",
        paradigm="graph generative model",
        covered_by_papers=["tabstruct-2026"],
        standardized_status="implemented",
        runnable_recommendation="yes",
        implementation_quality="medium",
        repository_url="https://github.com/amazon-science/tabsyn",
        notes=[
            "Integrated against the vendored TabSyn baseline implementation with local compatibility fixes for modern sklearn and Python packaging.",
            "End-to-end adult smoke validation now passes through the shared CLI.",
            "Still the most fragile of the implemented adapters because it depends on DGL, torch-geometric, and version-sensitive graph extensions.",
        ],
    ),
    "ctgan": ModelInventoryEntry(
        name="ctgan",
        family="gan",
        paradigm="conditional GAN",
        covered_by_papers=["tabstruct-2026", "tabula-2025"],
        standardized_status="implemented",
        runnable_recommendation="yes",
        implementation_quality="high",
        repository_url="https://github.com/sdv-dev/CTGAN",
        notes=[
            "Maintained open-source project with tests, packaging, and SDV ecosystem support.",
            "One of the cleanest non-diffusion baselines to integrate next.",
        ],
    ),
    "nflow": ModelInventoryEntry(
        name="nflow",
        family="flow",
        paradigm="normalizing flow",
        covered_by_papers=["tabstruct-2026"],
        standardized_status="implemented",
        runnable_recommendation="yes",
        implementation_quality="medium",
        repository_url="https://github.com/bayesiains/nflows",
        notes=[
            "Integrated as a direct nflows-backed masked autoregressive flow adapter rather than through Synthcity.",
            "End-to-end adult smoke validation now passes through the shared CLI.",
            "The benchmark label is generic, so the registry should continue documenting the exact flow architecture used in standardized runs.",
        ],
    ),
    "arf": ModelInventoryEntry(
        name="arf",
        family="tree",
        paradigm="adversarial random forest",
        covered_by_papers=["tabstruct-2026"],
        standardized_status="implemented",
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
        covered_by_papers=["tabstruct-2026"],
        standardized_status="implemented",
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
        covered_by_papers=["tabstruct-2026"],
        standardized_status="implemented",
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
        standardized_status="implemented",
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
        standardized_status="implemented",
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
}


def get_inventory_entry(model_name: str) -> ModelInventoryEntry:
    return MODEL_INVENTORY[model_name]


def list_inventory(*, benchmark: str | None = None) -> list[ModelInventoryEntry]:
    entries = sorted(MODEL_INVENTORY.values(), key=lambda entry: entry.name)
    if benchmark is None:
        return entries
    return [entry for entry in entries if benchmark in entry.covered_by_papers]


def list_inventory_names(*, benchmark: str | None = None) -> list[str]:
    return [entry.name for entry in list_inventory(benchmark=benchmark)]
