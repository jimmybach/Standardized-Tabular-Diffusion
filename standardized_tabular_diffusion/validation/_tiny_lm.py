"""Offline tiny causal-LM fixture used only by executable adapter validation."""

from __future__ import annotations

from pathlib import Path


def build_tiny_gpt2(path: Path, texts: list[str], *, seed: int = 1234) -> dict[str, int]:
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    path.mkdir(parents=True, exist_ok=True)
    backend = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        special_tokens=["[UNK]", "<|endoftext|>"],
        min_frequency=1,
    )
    backend.train_from_iterator(texts, trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        eos_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )
    tokenizer.padding_side = "left"
    tokenizer.save_pretrained(path)
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=128,
        n_ctx=128,
        n_embd=32,
        n_layer=1,
        n_head=1,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = GPT2LMHeadModel(config)
    model.save_pretrained(path, safe_serialization=True)
    return {"vocab_size": len(tokenizer), "parameters": sum(parameter.numel() for parameter in model.parameters())}


__all__ = ["build_tiny_gpt2"]
