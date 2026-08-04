from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, read_json
from standardized_tabular_diffusion.interfaces import ArtifactBundle, DatasetSpec, RunSpec
from standardized_tabular_diffusion.models._runtime import (
    SampleFileEvaluatorMixin,
    disable_torchvision_for_transformers,
)
from standardized_tabular_diffusion.models.base import BaseModelAdapter


class _TokenizedTextDataset:
    def __init__(self, tokenizer, texts: list[str], max_length: int) -> None:
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = tokenized["input_ids"]
        self.attention_mask = tokenized["attention_mask"]

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        input_ids = self.input_ids[idx]
        return {
            "input_ids": input_ids,
            "attention_mask": self.attention_mask[idx],
            "labels": input_ids.clone(),
        }


@dataclass
class _TabulaImports:
    AutoConfig: Any
    AutoModelForCausalLM: Any
    AutoTokenizer: Any
    Trainer: Any
    TrainingArguments: Any
    default_data_collator: Any


class TabulaAdapter(BaseModelAdapter, SampleFileEvaluatorMixin):
    model_name = "tabula"
    upstream_dirname = "."

    def _model_root(self, spec: RunSpec) -> Path:
        if spec.checkpoint_path is not None:
            return spec.checkpoint_path
        return spec.output_dir / "tabula_model"

    def _metadata_path(self, model_root: Path) -> Path:
        return model_root / "adapter_metadata.json"

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> pd.DataFrame:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("tabula requires dataset_spec.train_data_path")
        return pd.read_csv(dataset_spec.train_data_path)[dataset_spec.column_names].copy()

    def _limit_training_frame(self, train_df: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
        max_train_rows = spec.extra.get("max_train_rows")
        if max_train_rows is None or len(train_df) <= int(max_train_rows):
            return train_df
        return train_df.sample(n=int(max_train_rows), random_state=spec.seed).reset_index(drop=True)

    def _import_transformer_bits(self) -> _TabulaImports:
        with disable_torchvision_for_transformers():
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoTokenizer,
                Trainer,
                TrainingArguments,
                default_data_collator,
            )

        return _TabulaImports(
            AutoConfig=AutoConfig,
            AutoModelForCausalLM=AutoModelForCausalLM,
            AutoTokenizer=AutoTokenizer,
            Trainer=Trainer,
            TrainingArguments=TrainingArguments,
            default_data_collator=default_data_collator,
        )

    @staticmethod
    def _special_tokens() -> dict[str, str]:
        return {
            "column_sep": "<|col|>",
            "row_end": "<|endrow|>",
        }

    @staticmethod
    def _normalize_cell(value: Any) -> str:
        text = str(value)
        return text.replace("\n", " ").replace("\r", " ").strip()

    def _row_to_text(self, row: pd.Series, column_order: list[str], *, column_sep: str, row_end: str) -> str:
        cells = [f"{column}={self._normalize_cell(row[column])}" for column in column_order]
        return column_sep.join(cells) + row_end

    def _build_training_texts(
        self, train_df: pd.DataFrame, dataset_spec: DatasetSpec, *, column_sep: str, row_end: str
    ) -> list[str]:
        return [
            self._row_to_text(row, dataset_spec.column_names, column_sep=column_sep, row_end=row_end)
            for _, row in train_df.iterrows()
        ]

    def _build_tokenizer(self, imports: _TabulaImports, spec: RunSpec):
        tokenizer = imports.AutoTokenizer.from_pretrained(spec.extra.get("llm", "distilgpt2"))
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    self._special_tokens()["column_sep"],
                    self._special_tokens()["row_end"],
                ]
            }
        )
        return tokenizer

    def _build_model(self, imports: _TabulaImports, tokenizer, spec: RunSpec):
        model_name = spec.extra.get("llm", "distilgpt2")
        from_scratch = bool(spec.extra.get("from_scratch", False))
        if from_scratch:
            config = imports.AutoConfig.from_pretrained(model_name)
            for field in ("n_layer", "n_head", "n_embd", "n_positions", "n_ctx"):
                if spec.extra.get(field) is not None and hasattr(config, field):
                    setattr(config, field, int(spec.extra[field]))
            model = imports.AutoModelForCausalLM.from_config(config)
        else:
            model = imports.AutoModelForCausalLM.from_pretrained(model_name)
        model.resize_token_embeddings(len(tokenizer))
        return model

    @staticmethod
    def _training_args(imports: _TabulaImports, spec: RunSpec) -> Any:
        return imports.TrainingArguments(
            output_dir=str(spec.output_dir / "tabula_trainer"),
            overwrite_output_dir=True,
            num_train_epochs=float(spec.extra.get("epochs", 5)),
            per_device_train_batch_size=int(spec.extra.get("batch_size", 8)),
            gradient_accumulation_steps=int(spec.extra.get("gradient_accumulation_steps", 1)),
            learning_rate=float(spec.extra.get("learning_rate", 5e-5)),
            weight_decay=float(spec.extra.get("weight_decay", 0.0)),
            logging_steps=int(spec.extra.get("logging_steps", 50)),
            save_strategy="no",
            report_to=spec.extra.get("report_to", "none"),
            remove_unused_columns=False,
            no_cuda=spec.device == "cpu",
            seed=spec.seed,
        )

    def _build_training_metadata(
        self,
        train_df: pd.DataFrame,
        dataset_spec: DatasetSpec,
        tokenizer,
        texts: list[str],
        spec: RunSpec,
    ) -> dict[str, Any]:
        sample_size = min(len(texts), 256)
        token_lengths: list[int] = []
        for text in texts[:sample_size]:
            token_lengths.append(len(tokenizer(text)["input_ids"]))
        observed_max = max(token_lengths, default=64)
        observed_p95 = int(np.percentile(token_lengths, 95)) if token_lengths else observed_max
        recommended_max_length = max(128, min(2048, observed_p95 + 32))
        return {
            "column_names": list(dataset_spec.column_names),
            "numerical_columns": list(dataset_spec.numerical_columns),
            "categorical_columns": list(dataset_spec.categorical_columns),
            "target_columns": list(dataset_spec.target_columns),
            "task_type": dataset_spec.task_type,
            "column_sep": self._special_tokens()["column_sep"],
            "row_end": self._special_tokens()["row_end"],
            "recommended_start_col": dataset_spec.column_names[0],
            "recommended_temperature": float(spec.extra.get("temperature", 0.8)),
            "recommended_max_length": recommended_max_length,
            "observed_token_length_max": observed_max,
            "observed_token_length_p95": observed_p95,
            "from_scratch": bool(spec.extra.get("from_scratch", False)),
            "llm": spec.extra.get("llm", "distilgpt2"),
            "max_train_rows": None if spec.extra.get("max_train_rows") is None else int(spec.extra["max_train_rows"]),
        }

    @staticmethod
    def _load_training_metadata(model_root: Path) -> dict[str, Any]:
        metadata_path = model_root / "adapter_metadata.json"
        if not metadata_path.exists():
            return {}
        return read_json(metadata_path)

    @staticmethod
    def _coerce_value(column: str, value: str, metadata: dict[str, Any]) -> Any:
        if column in metadata.get("numerical_columns", []):
            return pd.to_numeric(value, errors="coerce")
        return value

    def _parse_generated_row(self, text: str, metadata: dict[str, Any]) -> dict[str, Any] | None:
        row_end = metadata["row_end"]
        column_sep = metadata["column_sep"]
        if row_end in text:
            text = text.split(row_end, 1)[0]
        pieces = [piece.strip() for piece in text.split(column_sep) if piece.strip()]
        values: dict[str, Any] = {}
        for piece in pieces:
            if "=" not in piece:
                continue
            column, raw_value = piece.split("=", 1)
            column = column.strip()
            raw_value = raw_value.strip()
            if column in metadata["column_names"] and column not in values:
                values[column] = self._coerce_value(column, raw_value, metadata)
        if any(column not in values for column in metadata["column_names"]):
            return None
        row = {column: values[column] for column in metadata["column_names"]}
        for column in metadata.get("numerical_columns", []):
            if pd.isna(row[column]):
                return None
        return row

    @staticmethod
    def _resolve_start_distribution(train_df: pd.DataFrame, start_col: str) -> dict[str, float] | list[Any]:
        series = train_df[start_col]
        if pd.api.types.is_numeric_dtype(series):
            return series.tolist()
        return series.astype(str).value_counts(normalize=True).to_dict()

    @staticmethod
    def _sample_start_value(start_dist: dict[str, float] | list[Any], seed: int) -> str:
        rng = np.random.default_rng(seed)
        if isinstance(start_dist, dict):
            keys = list(start_dist)
            probs = np.array([start_dist[key] for key in keys], dtype=float)
            probs = probs / probs.sum()
            return str(rng.choice(keys, p=probs))
        return str(start_dist[int(rng.integers(0, len(start_dist)))])

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        train_df = self._limit_training_frame(self._load_training_frame(dataset_spec), spec)
        imports = self._import_transformer_bits()
        tokenizer = self._build_tokenizer(imports, spec)
        model = self._build_model(imports, tokenizer, spec)
        texts = self._build_training_texts(
            train_df,
            dataset_spec,
            column_sep=self._special_tokens()["column_sep"],
            row_end=self._special_tokens()["row_end"],
        )
        max_length = int(spec.extra.get("max_length", 256))
        dataset = _TokenizedTextDataset(tokenizer, texts, max_length=max_length)
        trainer = imports.Trainer(
            model=model,
            args=self._training_args(imports, spec),
            train_dataset=dataset,
            data_collator=imports.default_data_collator,
        )
        trainer.train()
        model_root = self._model_root(spec)
        model_root.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(model_root))
        tokenizer.save_pretrained(str(model_root))
        metadata = self._build_training_metadata(train_df, dataset_spec, tokenizer, texts, spec)
        atomic_write_json(self._metadata_path(model_root), metadata)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            notes=[
                f"Saved TabuLa-compatible artifacts under {model_root}.",
                f"Recorded adapter metadata with recommended max_length={metadata['recommended_max_length']} and start_col={metadata['recommended_start_col']}.",
            ],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        import torch

        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        train_df = self._load_training_frame(dataset_spec)
        imports = self._import_transformer_bits()
        model_root = self._model_root(spec)
        metadata = self._load_training_metadata(model_root)
        trusted_model_root = self._validate_trusted_executable_artifact(
            spec,
            model_root,
            format_name="TabuLa model directory",
            allow_directory=True,
        )
        tokenizer = imports.AutoTokenizer.from_pretrained(str(trusted_model_root))
        model = imports.AutoModelForCausalLM.from_pretrained(str(trusted_model_root))
        if spec.device != "cpu" and torch.cuda.is_available():
            model = model.to(spec.device)
        model.eval()

        num_samples = spec.num_samples or len(train_df)
        start_col = spec.extra.get("start_col", metadata.get("recommended_start_col", dataset_spec.column_names[0]))
        start_dist = self._resolve_start_distribution(train_df, start_col)
        column_sep = metadata["column_sep"]
        row_end = metadata["row_end"]
        max_new_tokens = int(spec.extra.get("max_new_tokens", metadata.get("recommended_max_length", 256)))
        temperature = float(spec.extra.get("temperature", metadata.get("recommended_temperature", 0.8)))
        top_p = float(spec.extra.get("top_p", 0.95))
        max_tries = int(spec.extra.get("max_tries", max(100, num_samples * 10)))

        rows: list[dict[str, Any]] = []
        notes: list[str] = []
        attempts = 0
        while len(rows) < num_samples and attempts < max_tries:
            attempts += 1
            start_value = self._sample_start_value(start_dist, spec.seed + attempts)
            prompt = f"{start_col}={start_value}{column_sep}"
            tokenized = tokenizer(prompt, return_tensors="pt")
            normalized_inputs: dict[str, torch.Tensor] = {}
            for key, value in tokenized.items():
                tensor = value if isinstance(value, torch.Tensor) else torch.tensor([value], dtype=torch.long)
                normalized_inputs[key] = tensor.to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **normalized_inputs,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.convert_tokens_to_ids(row_end)
                    if hasattr(tokenizer, "convert_tokens_to_ids")
                    else None,
                )
            text = tokenizer.decode(generated[0], skip_special_tokens=False)
            parsed = self._parse_generated_row(text, metadata)
            if parsed is None:
                continue
            rows.append(parsed)

        if len(rows) < num_samples:
            raise RuntimeError(
                f"TabuLa sampling generated only {len(rows)} valid rows out of the requested {num_samples}. "
                "Try increasing epochs, max_new_tokens, or max_tries."
            )

        sample_df = pd.DataFrame(rows, columns=dataset_spec.column_names)
        sample_path = spec.output_dir / "samples.csv"
        self._write_dataframe_csv(sample_df, sample_path)
        notes.append(
            f"TabuLa sampling succeeded with start_col={start_col}, temperature={temperature}, top_p={top_p}, max_new_tokens={max_new_tokens}, attempts={attempts}."
        )
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            generated_sample_path=sample_path,
            notes=notes,
        )
        return self._write_bundle(bundle)

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        return self._evaluate_from_sample_file(spec)
