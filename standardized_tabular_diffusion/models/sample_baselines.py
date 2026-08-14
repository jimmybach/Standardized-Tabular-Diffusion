from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from standardized_tabular_diffusion.evaluation.serialization import (
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
)
from standardized_tabular_diffusion.interfaces import ArtifactBundle, DatasetSpec, RunSpec
from standardized_tabular_diffusion.models._runtime import SampleFileEvaluatorMixin
from standardized_tabular_diffusion.models.base import BaseModelAdapter
from standardized_tabular_diffusion.output_decoding import decode_declared_integer_columns
from standardized_tabular_diffusion.runtime_contracts import require_cpu_device


class _OfficialCTGANPackageAdapter(BaseModelAdapter, SampleFileEvaluatorMixin):
    checkpoint_filename = "model.pkl"
    _OFFICIAL_PACKAGE_VERSION = "0.12.1"

    def _resolve_checkpoint_path(self, spec: RunSpec) -> Path:
        if spec.checkpoint_path is not None:
            return spec.checkpoint_path
        return spec.output_dir / self.checkpoint_filename

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> pd.DataFrame:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError(f"{self.model_name} requires dataset_spec.train_data_path")
        frame = pd.read_csv(dataset_spec.train_data_path)
        return frame[dataset_spec.column_names].copy()

    def _discrete_columns(self, dataset_spec: DatasetSpec) -> list[str]:
        discrete = list(dataset_spec.categorical_columns)
        if dataset_spec.task_type == "classification":
            for target in dataset_spec.target_columns:
                if target not in discrete:
                    discrete.append(target)
        return discrete

    def _train_kwargs(self, spec: RunSpec) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if spec.extra.get("epochs") is not None:
            kwargs["epochs"] = int(spec.extra["epochs"])
        if spec.extra.get("batch_size") is not None:
            kwargs["batch_size"] = int(spec.extra["batch_size"])
        if spec.extra.get("verbose") is not None:
            kwargs["verbose"] = bool(spec.extra["verbose"])
        if self.model_name == "ctgan":
            for name in ("embedding_dim", "discriminator_steps", "pac"):
                if spec.extra.get(name) is not None:
                    kwargs[name] = int(spec.extra[name])
            for name in ("generator_lr", "generator_decay", "discriminator_lr", "discriminator_decay"):
                if spec.extra.get(name) is not None:
                    kwargs[name] = float(spec.extra[name])
            if spec.extra.get("log_frequency") is not None:
                kwargs["log_frequency"] = bool(spec.extra["log_frequency"])
            for name in ("generator_dim", "discriminator_dim"):
                if spec.extra.get(name) is not None:
                    kwargs[name] = tuple(int(value) for value in spec.extra[name])
            batch_size = int(kwargs.get("batch_size", 500))
            pac = int(kwargs.get("pac", 10))
            epochs = int(kwargs.get("epochs", 300))
            embedding_dim = int(kwargs.get("embedding_dim", 128))
            discriminator_steps = int(kwargs.get("discriminator_steps", 1))
            dimensions = (
                tuple(kwargs.get("generator_dim", (256, 256))),
                tuple(kwargs.get("discriminator_dim", (256, 256))),
            )
            if batch_size <= 0 or batch_size % 2 or pac <= 0 or batch_size % pac:
                raise ValueError("CTGAN batch_size must be positive, even, and divisible by positive pac")
            if epochs <= 0 or embedding_dim <= 0 or discriminator_steps <= 0:
                raise ValueError("CTGAN epochs, embedding_dim, and discriminator_steps must be positive")
            if any(not values or any(value <= 0 for value in values) for values in dimensions):
                raise ValueError("CTGAN generator_dim and discriminator_dim must contain positive integers")
        elif self.model_name == "tvae":
            for name in ("embedding_dim",):
                if spec.extra.get(name) is not None:
                    kwargs[name] = int(spec.extra[name])
            for name in ("l2scale", "loss_factor"):
                if spec.extra.get(name) is not None:
                    kwargs[name] = float(spec.extra[name])
            for name in ("compress_dims", "decompress_dims"):
                if spec.extra.get(name) is not None:
                    kwargs[name] = tuple(int(value) for value in spec.extra[name])
            batch_size = int(kwargs.get("batch_size", 500))
            epochs = int(kwargs.get("epochs", 300))
            embedding_dim = int(kwargs.get("embedding_dim", 128))
            l2scale = float(kwargs.get("l2scale", 1e-5))
            loss_factor = float(kwargs.get("loss_factor", 2))
            dimensions = (
                tuple(kwargs.get("compress_dims", (128, 128))),
                tuple(kwargs.get("decompress_dims", (128, 128))),
            )
            if batch_size <= 0 or epochs <= 0 or embedding_dim <= 0 or loss_factor <= 0:
                raise ValueError("TVAE batch_size, epochs, embedding_dim, and loss_factor must be positive")
            if l2scale < 0:
                raise ValueError("TVAE l2scale must be non-negative")
            if any(not values or any(value <= 0 for value in values) for values in dimensions):
                raise ValueError("TVAE compress_dims and decompress_dims must contain positive integers")
        return kwargs

    def _import_synthesizer_cls(self):
        try:
            installed_version = version("ctgan")
        except PackageNotFoundError as exc:
            raise ModuleNotFoundError(
                f"{self.model_name.upper()} requires the pinned official package; install "
                f"`standardized-tabular-diffusion[{self.model_name}]`."
            ) from exc
        if installed_version != self._OFFICIAL_PACKAGE_VERSION:
            raise RuntimeError(
                f"{self.model_name.upper()} package version mismatch: "
                f"expected {self._OFFICIAL_PACKAGE_VERSION}, observed {installed_version}. "
                f"Install `standardized-tabular-diffusion[{self.model_name}]` to restore the validated runtime."
            )
        if self.model_name == "ctgan":
            from ctgan import CTGAN  # pylint: disable=import-error

            return CTGAN
        if self.model_name == "tvae":
            from ctgan import TVAE  # pylint: disable=import-error

            return TVAE
        raise RuntimeError(f"Unsupported official ctgan-package adapter: {self.model_name}")

    def _build_synthesizer(self, spec: RunSpec):
        synthesizer_cls = self._import_synthesizer_cls()
        kwargs = self._train_kwargs(spec)
        if self.model_name == "tvae" and spec.device.startswith("cuda:") and spec.device != "cuda:0":
            raise ValueError(
                "Official TVAE selects the default visible CUDA device during fit; "
                "use 'cuda' or 'cuda:0', or remap devices with CUDA_VISIBLE_DEVICES"
            )
        kwargs["enable_gpu"] = spec.device.startswith("cuda")
        model = synthesizer_cls(**kwargs)
        if self.model_name == "ctgan":
            model.set_device(spec.device)
        return model

    def _save_model(self, model: Any, checkpoint_path: Path) -> None:
        model.save(checkpoint_path)

    def _load_model(self, spec: RunSpec, checkpoint_path: Path) -> Any:
        trusted_checkpoint = self._validate_trusted_executable_artifact(
            spec,
            checkpoint_path,
            format_name="PyTorch/pickle",
        )
        synthesizer_cls = self._import_synthesizer_cls()
        model = synthesizer_cls.load(trusted_checkpoint)
        model.set_device(spec.device)
        return model

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        train_df = self._load_training_frame(dataset_spec)
        missing_counts = train_df.isna().sum()
        if bool(missing_counts.any()):
            observed = {str(column): int(count) for column, count in missing_counts.items() if count}
            raise ValueError(
                f"{self.model_name} does not accept missing values in this benchmark. "
                f"Run the explicit train-fitted preprocessing module first; observed: {observed}"
            )
        model = self._build_synthesizer(spec)
        model.set_random_state(spec.seed)
        model.fit(train_df, discrete_columns=self._discrete_columns(dataset_spec))
        checkpoint_path = self._resolve_checkpoint_path(spec)
        self._save_model(model, checkpoint_path)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            notes=[
                f"Serialized {self.model_name} checkpoint written to {checkpoint_path}.",
                *(
                    [f"Official ctgan package version: {self._OFFICIAL_PACKAGE_VERSION}."]
                    if self.model_name in {"ctgan", "tvae"}
                    else []
                ),
            ],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        checkpoint_path = self._resolve_checkpoint_path(spec)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {self.model_name}: {checkpoint_path}")

        model = self._load_model(spec, checkpoint_path)
        model.set_random_state(spec.seed)

        train_df = self._load_training_frame(dataset_spec)
        num_samples = spec.num_samples or len(train_df)
        sample_df = model.sample(num_samples)
        sample_df = sample_df[dataset_spec.column_names].copy()
        if len(sample_df) != num_samples or bool(sample_df.isna().any().any()):
            raise ValueError(
                f"Official {self.model_name} returned an invalid row count or missing sample values."
            )
        native_sample_df = sample_df.copy()
        sample_df, integer_decoding = decode_declared_integer_columns(
            sample_df,
            dataset_spec,
            model_name=self.model_name.upper(),
        )
        native_sample_path = None
        if any(record["changed_rows"] for record in integer_decoding.values()):
            native_sample_path = spec.output_dir / f"{self.model_name}_native_samples.csv"
            self._write_dataframe_csv(native_sample_df, native_sample_path)
        sample_path = spec.output_dir / "samples.csv"
        self._write_dataframe_csv(sample_df, sample_path)
        if integer_decoding:
            atomic_write_json(
                spec.output_dir / f"{self.model_name}_sample_metadata.json",
                {
                    "package": "ctgan",
                    "package_version": self._OFFICIAL_PACKAGE_VERSION,
                    "model": self.model_name,
                    "seed": spec.seed,
                    "requested_rows": num_samples,
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "sample_path": str(sample_path),
                    "sample_sha256": sha256_file(sample_path),
                    "integer_decoding": integer_decoding,
                    "native_sample_path": str(native_sample_path),
                    "native_sample_sha256": sha256_file(native_sample_path),
                },
            )
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            generated_sample_path=sample_path,
            notes=[
                "Declared integer columns are decoded with numpy.rint after official package sampling; "
                "the unmodified official output is retained whenever values change."
            ],
        )
        return self._write_bundle(bundle)

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        return self._evaluate_from_sample_file(spec)


class CTGANAdapter(_OfficialCTGANPackageAdapter):
    model_name = "ctgan"
    upstream_dirname = "."


class TVAEAdapter(_OfficialCTGANPackageAdapter):
    model_name = "tvae"
    upstream_dirname = "."


class SMOTEAdapter(BaseModelAdapter, SampleFileEvaluatorMixin):
    model_name = "smote"
    upstream_dirname = "."
    package_name = "imbalanced-learn"
    package_version = "0.14.2"

    def _load_official_samplers(self) -> tuple[type[Any], type[Any], type[Any]]:
        try:
            observed_version = version(self.package_name)
        except PackageNotFoundError as exc:
            raise ImportError(
                "SMOTE requires the optional official imbalanced-learn package. "
                "Install the project with the 'smote' extra."
            ) from exc
        if observed_version != self.package_version:
            raise ImportError(
                "SMOTE requires the exact validated official package version "
                f"{self.package_name}=={self.package_version}; observed {observed_version}."
            )

        from imblearn.over_sampling import SMOTE, SMOTEN, SMOTENC

        return SMOTE, SMOTENC, SMOTEN

    def _create_sampler(
        self,
        *,
        feature_columns: list[str],
        categorical_columns: list[str],
        random_state: int,
        k_neighbors: int,
        sampling_strategy: Any,
    ) -> tuple[Any, str]:
        smote_class, smotenc_class, smoten_class = self._load_official_samplers()
        common = {
            "sampling_strategy": sampling_strategy,
            "random_state": random_state,
            "k_neighbors": k_neighbors,
        }
        if not categorical_columns:
            return smote_class(**common), "SMOTE"
        if len(categorical_columns) == len(feature_columns):
            return smoten_class(**common), "SMOTEN"
        return (
            smotenc_class(categorical_features=categorical_columns, **common),
            "SMOTENC",
        )

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> pd.DataFrame:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("SMOTE requires dataset_spec.train_data_path")
        frame = pd.read_csv(dataset_spec.train_data_path)
        return frame[dataset_spec.column_names].copy()

    def train(self, spec: RunSpec) -> ArtifactBundle:
        require_cpu_device(self.model_name, spec.device)
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        if dataset_spec.task_type != "classification":
            raise ValueError("SMOTE is only supported for classification datasets.")
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            notes=[
                "SMOTE has no persistent fitted checkpoint in the standardized adapter.",
                "Sample generation recomputes the resampled table directly from the canonical training split.",
            ],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        require_cpu_device(self.model_name, spec.device)
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        if dataset_spec.task_type != "classification":
            raise ValueError("SMOTE is only supported for classification datasets.")

        train_df = self._load_training_frame(dataset_spec)
        if len(dataset_spec.target_columns) != 1:
            raise ValueError("SMOTE requires exactly one target column.")
        target = dataset_spec.target_columns[0]
        feature_cols = [column for column in dataset_spec.column_names if column != target]
        if not feature_cols:
            raise ValueError("SMOTE requires at least one feature column.")
        missing_counts = train_df[[*feature_cols, target]].isna().sum()
        if bool(missing_counts.any()):
            observed = {str(column): int(count) for column, count in missing_counts.items() if count}
            raise ValueError(
                "SMOTE does not accept missing values in this benchmark. "
                f"Run the explicit preprocessing module first; observed: {observed}"
            )

        x_train = train_df[feature_cols].copy()
        y_train = train_df[target]
        k_neighbors = int(spec.extra.get("k_neighbors", 5))
        if k_neighbors < 1:
            raise ValueError(f"SMOTE k_neighbors must be positive; observed {k_neighbors}.")
        sampling_strategy = spec.extra.get("sampling_strategy", "auto")
        if callable(sampling_strategy) or not isinstance(sampling_strategy, (str, float, dict)):
            raise TypeError(
                "SMOTE sampling_strategy must be a JSON-serializable string, float, or dictionary."
            )
        try:
            json.dumps(sampling_strategy, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "SMOTE sampling_strategy must contain only finite JSON-serializable values."
            ) from exc
        class_counts = y_train.value_counts(dropna=False)
        if len(class_counts) < 2:
            raise ValueError("SMOTE requires at least two target classes.")
        if int(class_counts.min()) <= k_neighbors:
            raise ValueError(
                f"SMOTE k_neighbors={k_neighbors} requires at least {k_neighbors + 1} rows in every class; "
                f"smallest class has {int(class_counts.min())}."
            )

        categorical_columns = [column for column in feature_cols if column in dataset_spec.categorical_columns]
        categorical_indices = [feature_cols.index(column) for column in categorical_columns]
        sampler, sampler_name = self._create_sampler(
            feature_columns=feature_cols,
            categorical_columns=categorical_columns,
            random_state=spec.seed,
            k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
        )
        x_resampled, y_resampled = sampler.fit_resample(x_train, y_train)
        sampled_df = pd.DataFrame(x_resampled, columns=feature_cols)
        sampled_df[target] = y_resampled
        sampled_df = sampled_df[dataset_spec.column_names]

        desired_rows = len(sampled_df) if spec.num_samples is None else spec.num_samples
        if desired_rows < 1:
            raise ValueError(f"SMOTE num_samples must be positive; observed {desired_rows}.")
        replace = len(sampled_df) < desired_rows
        sampled_df = sampled_df.sample(n=desired_rows, replace=replace, random_state=spec.seed).reset_index(drop=True)

        native_sample_df = sampled_df.copy()
        sampled_df, integer_decoding = decode_declared_integer_columns(
            sampled_df,
            dataset_spec,
            model_name="SMOTE",
        )
        native_sample_path = None
        if any(record["changed_rows"] for record in integer_decoding.values()):
            native_sample_path = spec.output_dir / "smote_native_samples.csv"
            atomic_write_bytes(native_sample_path, native_sample_df.to_csv(index=False).encode("utf-8"))

        sample_path = spec.output_dir / "samples.csv"
        atomic_write_bytes(sample_path, sampled_df.to_csv(index=False).encode("utf-8"))
        sample_metadata = {
            "sampler": sampler_name,
            "package": self.package_name,
            "package_version": self.package_version,
            "random_state": spec.seed,
            "k_neighbors": k_neighbors,
            "sampling_strategy": sampling_strategy,
            "categorical_columns": categorical_columns,
            "categorical_indices": categorical_indices,
            "source_rows": len(train_df),
            "balanced_rows": len(x_resampled),
            "output_rows": len(sampled_df),
        }
        if integer_decoding:
            sample_metadata.update(
                {
                    "integer_decoding": integer_decoding,
                    "native_sample_path": None if native_sample_path is None else str(native_sample_path),
                    "native_sample_sha256": (
                        None if native_sample_path is None else sha256_file(native_sample_path)
                    ),
                }
            )
        atomic_write_json(spec.output_dir / "smote_metadata.json", sample_metadata)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            generated_sample_path=sample_path,
            notes=[
                "Rows are drawn from the SMOTE-resampled table.",
                f"Mixed-type handling used {sampler_name}; categorical values were never interpolated as continuous data.",
                "Declared integer columns are decoded with numpy.rint at the adapter boundary; "
                "the unmodified official interpolation output is retained whenever values change.",
                "If num_samples differs from the balanced-resample size, the adapter resamples rows from the SMOTE output.",
            ],
        )
        return self._write_bundle(bundle)

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        return self._evaluate_from_sample_file(spec)
