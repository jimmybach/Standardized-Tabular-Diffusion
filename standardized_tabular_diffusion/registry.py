from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

from standardized_tabular_diffusion.datasets import discover_dataset_specs

if TYPE_CHECKING:
    from standardized_tabular_diffusion.models.base import BaseModelAdapter


class AdapterDependencyError(ImportError):
    """Raised when a requested adapter is missing an optional runtime dependency."""


class AdapterSourceUnavailableError(FileNotFoundError):
    """Raised when a source-backed adapter is used without its audited source tree."""


class AdapterValidationLevel(StrEnum):
    REGISTERED = "registered"
    ADAPTER_COMPLETE = "adapter-complete"
    SMOKE_VALIDATED = "smoke-validated"
    NATIVE_PARITY_VALIDATED = "native-parity-validated"


@dataclass(frozen=True)
class AdapterSpec:
    module: str
    class_name: str
    source_authority: str
    distribution_form: str
    reproduction_target: str
    modification_status: str
    validation_level: AdapterValidationLevel = AdapterValidationLevel.ADAPTER_COMPLETE
    benchmark_track: str = "experimental"
    support_level: str = "unsupported"
    actions: tuple[str, ...] = ("train", "sample", "evaluate")
    task_types: tuple[str, ...] = ("classification", "regression")
    requires_dataset_paths: bool = True
    evaluation_input: str = "sample-file"
    install_extra: str = "models"
    source_root: str | None = None
    upstream_repository: str | None = None
    upstream_revision: str | None = None
    patch_set_ids: tuple[str, ...] = ()
    license_status: str = "source-license-present; transitive-review-pending"
    revision_status: str = "unresolved"
    evidence_records: tuple[str, ...] = ()

    def to_dict(self, model_name: str) -> dict[str, Any]:
        payload = asdict(self)
        payload["model_id"] = model_name
        payload["validation_level"] = self.validation_level.value
        payload["actions"] = list(self.actions)
        payload["task_types"] = list(self.task_types)
        payload["evidence_records"] = list(self.evidence_records)
        payload["patch_set_ids"] = list(self.patch_set_ids)
        return payload


def _spec(
    module: str,
    class_name: str,
    *,
    authority: str,
    distribution: str,
    target: str = "original-method",
    modification: str = "adapter-only",
    install_extra: str = "models",
    evaluation_input: str = "sample-file",
    task_types: tuple[str, ...] = ("classification", "regression"),
    requires_dataset_paths: bool = True,
    source_root: str | None = None,
    upstream_repository: str | None = None,
    upstream_revision: str | None = None,
    revision_status: str = "unresolved",
    patch_set_ids: tuple[str, ...] = (),
    evidence_records: tuple[str, ...] = (),
    license_status: str = "source-license-present; transitive-review-pending",
    validation_level: AdapterValidationLevel = AdapterValidationLevel.ADAPTER_COMPLETE,
) -> AdapterSpec:
    return AdapterSpec(
        module=module,
        class_name=class_name,
        source_authority=authority,
        distribution_form=distribution,
        reproduction_target=target,
        modification_status=modification,
        install_extra=install_extra,
        evaluation_input=evaluation_input,
        task_types=task_types,
        requires_dataset_paths=requires_dataset_paths,
        source_root=source_root,
        upstream_repository=upstream_repository,
        upstream_revision=upstream_revision,
        revision_status=revision_status,
        patch_set_ids=patch_set_ids,
        evidence_records=evidence_records,
        license_status=license_status,
        validation_level=validation_level,
    )


# This registry is deliberately conservative: mocked contract tests establish
# adapter completeness only. An adapter is promoted beyond that level only with
# a retained evidence record; benchmark eligibility and release support remain
# independent gates.
_ADAPTER_SPECS: dict[str, AdapterSpec] = {
    "arf": _spec(
        "standardized_tabular_diffusion.models.final_wave_baselines",
        "ARFAdapter",
        authority="third-party",
        distribution="package",
    ),
    "bn": _spec(
        "standardized_tabular_diffusion.models.structured_baselines",
        "BNAdapter",
        authority="local",
        distribution="hybrid",
        modification="semantic-patched",
    ),
    "codi": _spec(
        "standardized_tabular_diffusion.models.vendored_baselines",
        "CoDiAdapter",
        authority="benchmark-vendored",
        distribution="source",
        target="benchmark-snapshot",
        modification="compatibility-patched",
        source_root="TabSyn-main",
    ),
    "ctab-gan": _spec(
        "standardized_tabular_diffusion.models.vendored_baselines",
        "CTABGANAdapter",
        authority="benchmark-vendored",
        distribution="source",
        target="benchmark-snapshot",
        modification="compatibility-patched",
        source_root="TabDDPM-main",
    ),
    "ctab-gan-plus": _spec(
        "standardized_tabular_diffusion.models.next_wave_baselines",
        "CTABGANPlusAdapter",
        authority="method-author",
        distribution="source",
        modification="compatibility-patched",
        source_root="TabDDPM-main",
    ),
    "ctgan": _spec(
        "standardized_tabular_diffusion.models.sample_baselines",
        "CTGANAdapter",
        authority="method-author",
        distribution="package",
        install_extra="ctgan",
        upstream_repository="https://github.com/sdv-dev/CTGAN",
        upstream_revision="826da23f8f9385ad15fd206ecad691e04cb0ccdc",
        revision_status="pinned-official-package-native-parity-validated",
        license_status="BUSL-1.1 package dependency; official-track and release legal review required",
        evidence_records=(
            "docs/UPSTREAM_SOURCE_AUDIT.md",
            "docs/CTGAN_VALIDATION.md",
            "docs/evidence/ctgan/native-parity-run-30910275922.json",
            "standardized_tabular_diffusion/resources/upstream/source-lock.json",
            ".github/workflows/ctgan-validation.yml",
        ),
        validation_level=AdapterValidationLevel.NATIVE_PARITY_VALIDATED,
    ),
    "goggle": _spec(
        "standardized_tabular_diffusion.models.structured_baselines",
        "GoggleAdapter",
        authority="benchmark-vendored",
        distribution="source",
        target="benchmark-snapshot",
        modification="compatibility-patched",
        source_root="TabSyn-main",
    ),
    "great": _spec(
        "standardized_tabular_diffusion.models.final_wave_baselines",
        "GReaTAdapter",
        authority="benchmark-vendored",
        distribution="source",
        target="benchmark-snapshot",
        modification="semantic-patched",
        source_root="TabSyn-main",
    ),
    "nrgboost": _spec(
        "standardized_tabular_diffusion.models.next_wave_baselines",
        "NRGBoostAdapter",
        authority="method-author",
        distribution="package",
    ),
    "nflow": _spec(
        "standardized_tabular_diffusion.models.structured_baselines",
        "NFlowAdapter",
        authority="local",
        distribution="hybrid",
        modification="semantic-patched",
    ),
    "realtabformer": _spec(
        "standardized_tabular_diffusion.models.next_wave_baselines",
        "REaLTabFormerAdapter",
        authority="method-author",
        distribution="package",
    ),
    "smote": _spec(
        "standardized_tabular_diffusion.models.sample_baselines",
        "SMOTEAdapter",
        authority="third-party",
        distribution="package",
        target="classical-oversampling-reference",
        install_extra="smote",
        task_types=("classification",),
        upstream_repository="https://github.com/scikit-learn-contrib/imbalanced-learn",
        upstream_revision="8504e95f0160f61d1b617ca66f779646d2ee609e",
        revision_status="pinned-canonical-package-native-parity-pending",
        license_status="MIT package dependency; classification-only reference baseline",
        evidence_records=(
            "docs/UPSTREAM_SOURCE_AUDIT.md",
            "docs/SMOTE_VALIDATION.md",
            "standardized_tabular_diffusion/resources/upstream/source-lock.json",
            ".github/workflows/smote-validation.yml",
        ),
    ),
    "stasy": _spec(
        "standardized_tabular_diffusion.models.vendored_baselines",
        "STaSyAdapter",
        authority="benchmark-vendored",
        distribution="source",
        target="benchmark-snapshot",
        modification="compatibility-patched",
        source_root="TabSyn-main",
    ),
    "tabsds": _spec(
        "standardized_tabular_diffusion.models.paper_gap_baselines",
        "TabSDSAdapter",
        authority="local",
        distribution="source",
        modification="semantic-patched",
    ),
    "tabebm": _spec(
        "standardized_tabular_diffusion.models.final_wave_baselines",
        "TabEBMAdapter",
        authority="local",
        distribution="hybrid",
        modification="semantic-patched",
        task_types=("classification",),
    ),
    "tabdiff": _spec(
        "standardized_tabular_diffusion.models.tabdiff",
        "TabDiffAdapter",
        authority="method-author",
        distribution="source",
        modification="adapter-only",
        install_extra="evaluation",
        source_root="TabDiff-main",
        upstream_repository="https://github.com/MinkaiXu/TabDiff",
        upstream_revision="5ecdb3356261aea72716cc9a779f31d7ad083bf4",
        revision_status="pinned-exact-native-parity-validated",
        validation_level=AdapterValidationLevel.NATIVE_PARITY_VALIDATED,
        evidence_records=(
            "docs/UPSTREAM_SOURCE_AUDIT.md",
            "docs/TABDIFF_VALIDATION.md",
            "docs/evidence/tabdiff/native-parity-run-30866879879.json",
            "standardized_tabular_diffusion/resources/upstream/source-lock.json",
            "standardized_tabular_diffusion/resources/upstream/tabdiff-source-manifest.json",
            ".github/workflows/tabdiff-validation.yml",
        ),
    ),
    "tabddpm": _spec(
        "standardized_tabular_diffusion.models.tabddpm",
        "TabDDPMAdapter",
        authority="method-author",
        distribution="source",
        modification="adapter-only",
        install_extra="evaluation",
        evaluation_input="upstream-artifacts",
        requires_dataset_paths=False,
        source_root="TabDDPM-main",
        upstream_repository="https://github.com/yandex-research/tab-ddpm",
        upstream_revision="b476257dd460b778ba09eb97f7a51d6490fa17f8",
        revision_status="pinned-complete-native-parity-validated",
        validation_level=AdapterValidationLevel.NATIVE_PARITY_VALIDATED,
        evidence_records=(
            "docs/UPSTREAM_SOURCE_AUDIT.md",
            "docs/TABDDPM_VALIDATION.md",
            "docs/evidence/tabddpm/native-parity-run-30863212268.json",
            "standardized_tabular_diffusion/resources/upstream/source-lock.json",
            "standardized_tabular_diffusion/resources/upstream/tabddpm-source-manifest.json",
            ".github/workflows/tabddpm-validation.yml",
        ),
    ),
    "tabularargn": _spec(
        "standardized_tabular_diffusion.models.paper_gap_baselines",
        "TabularARGNAdapter",
        authority="method-author",
        distribution="package",
    ),
    "tabula": _spec(
        "standardized_tabular_diffusion.models.tabula",
        "TabulaAdapter",
        authority="local",
        distribution="source",
        modification="semantic-patched",
    ),
    "tabsyn": _spec(
        "standardized_tabular_diffusion.models.tabsyn",
        "TabSynAdapter",
        authority="method-author",
        distribution="source",
        modification="adapter-only",
        install_extra="evaluation",
        source_root="TabSyn-main",
        upstream_repository="https://github.com/amazon-science/tabsyn",
        upstream_revision="cb5ac0f74ec36ee88e7a974a393dfbef50d42da7",
        revision_status="pinned-exact-native-parity-validated",
        validation_level=AdapterValidationLevel.NATIVE_PARITY_VALIDATED,
        evidence_records=(
            "docs/UPSTREAM_SOURCE_AUDIT.md",
            "docs/TABSYN_VALIDATION.md",
            "docs/evidence/tabsyn/native-parity-run-30871758645.json",
            "standardized_tabular_diffusion/resources/upstream/source-lock.json",
            "standardized_tabular_diffusion/resources/upstream/tabsyn-source-manifest.json",
            ".github/workflows/tabsyn-validation.yml",
        ),
    ),
    "tvae": _spec(
        "standardized_tabular_diffusion.models.sample_baselines",
        "TVAEAdapter",
        authority="method-author",
        distribution="package",
        install_extra="tvae",
        upstream_repository="https://github.com/sdv-dev/CTGAN",
        upstream_revision="826da23f8f9385ad15fd206ecad691e04cb0ccdc",
        revision_status="pinned-official-package-native-parity-validated",
        license_status="BUSL-1.1 package dependency; official-track and release legal review required",
        evidence_records=(
            "docs/UPSTREAM_SOURCE_AUDIT.md",
            "docs/TVAE_VALIDATION.md",
            "docs/evidence/tvae/native-parity-run-30913867621.json",
            "standardized_tabular_diffusion/resources/upstream/source-lock.json",
            ".github/workflows/tvae-validation.yml",
        ),
        validation_level=AdapterValidationLevel.NATIVE_PARITY_VALIDATED,
    ),
}


def get_adapter_spec(model_name: str) -> AdapterSpec:
    try:
        return _ADAPTER_SPECS[model_name]
    except KeyError as exc:
        available = ", ".join(_ADAPTER_SPECS)
        raise KeyError(f"Unknown model adapter {model_name!r}. Available adapters: {available}") from exc


def list_adapter_specs() -> dict[str, dict[str, Any]]:
    return {name: spec.to_dict(name) for name, spec in _ADAPTER_SPECS.items()}


def _load_adapter_class(model_name: str) -> type[BaseModelAdapter]:
    spec = get_adapter_spec(model_name)
    try:
        module = import_module(spec.module)
    except ModuleNotFoundError as exc:
        missing = exc.name or "an optional dependency"
        if missing.startswith("standardized_tabular_diffusion"):
            raise
        raise AdapterDependencyError(
            f"Adapter {model_name!r} cannot import optional dependency {missing!r}. "
            f"The common {spec.install_extra!r} extra can be installed with "
            f'`pip install "standardized-tabular-diffusion[{spec.install_extra}]"`; '
            "model-specific dependencies may still be required, so also follow the model setup documentation."
        ) from exc

    adapter_class: Any = getattr(module, spec.class_name)
    return adapter_class


def get_adapter(model_name: str, repo_root: Path | None = None) -> BaseModelAdapter:
    resolved_root = repo_root or Path(__file__).resolve().parents[1]
    spec = get_adapter_spec(model_name)
    if spec.source_root is not None and not (resolved_root / spec.source_root).is_dir():
        raise AdapterSourceUnavailableError(
            f"Adapter {model_name!r} requires the audited {spec.source_root!r} source tree, but it is absent under "
            f"{resolved_root}. The lightweight wheel intentionally does not bundle upstream source snapshots; "
            "use a complete repository checkout whose source revision and licenses have been reviewed."
        )
    return _load_adapter_class(model_name)(resolved_root)


def list_models() -> list[str]:
    return list(_ADAPTER_SPECS)


def list_datasets(repo_root: Path | None = None) -> list[str]:
    return list(discover_dataset_specs(repo_root=repo_root).keys())


__all__ = [
    "AdapterDependencyError",
    "AdapterSpec",
    "AdapterSourceUnavailableError",
    "AdapterValidationLevel",
    "get_adapter",
    "get_adapter_spec",
    "list_adapter_specs",
    "list_datasets",
    "list_models",
]
