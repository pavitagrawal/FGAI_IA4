"""Reproducible English→Assamese MT experiment utilities.

The notebook is the public entry point.  This module keeps the substantial code
testable and prevents state hidden in notebook cells from changing the results.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import random
import re
import time
import unicodedata
import urllib.request
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


DATA_URL = "https://www.manythings.org/anki/asm-eng.zip"
DATA_PAGE = "https://www.manythings.org/anki/"
LICENSE = "CC BY 2.0 FR (Tatoeba; retain per-sentence attribution)"


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    profile: str = "quick"
    max_train: int | None = 2500
    max_validation: int = 400
    max_test: int = 500
    max_source_tokens: int = 64
    max_target_tokens: int = 64
    basic_vocab_size: int = 3000
    basic_d_model: int = 256
    basic_heads: int = 8
    basic_layers: int = 4
    basic_ffn: int = 1024
    basic_dropout: float = 0.1
    basic_epochs: int = 8
    basic_batch_size: int = 64
    basic_lr: float = 3e-4
    pretrained_epochs: int = 1
    pretrained_batch_size: int = 4
    grad_accum_steps: int = 4
    pretrained_lr: float = 2e-4
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    inference_batch_size: int = 1
    num_beams: int = 1
    max_new_tokens: int = 72
    mt5_model: str = "google/mt5-small"
    llama_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    gemma_model: str = "google/gemma-3-270m-it"

    @classmethod
    def full(cls) -> "ExperimentConfig":
        return cls(
            profile="full",
            max_train=None,
            max_validation=700,
            max_test=700,
            basic_vocab_size=4000,
            basic_d_model=320,
            basic_layers=6,
            basic_ffn=1280,
            basic_epochs=25,
            basic_batch_size=64,
            pretrained_epochs=3,
            pretrained_batch_size=8,
            grad_accum_steps=2,
            pretrained_lr=1e-4,
            lora_r=16,
            lora_alpha=32,
            num_beams=1,
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFC", str(value))
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Cc" or ch in "\t\n")
    return re.sub(r"\s+", " ", value).strip()


def leakage_key(value: str) -> str:
    value = unicodedata.normalize("NFKC", normalize_text(value)).casefold()
    value = "".join(ch for ch in value if not unicodedata.category(ch).startswith("P"))
    return re.sub(r"\s+", " ", value).strip()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 MT-assignment"})
    with urllib.request.urlopen(request, timeout=90) as response, destination.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)


def prepare_dataset(
    root: str | Path,
    config: ExperimentConfig,
    archive_override: str | Path | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Download, clean, group-split and persist the Tatoeba pairs.

    Splitting is by normalized English source, so alternative translations of the
    same source can never leak across train/validation/test.
    """
    root = Path(root)
    raw_dir, processed_dir = root / "data" / "raw", root / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    archive = Path(archive_override) if archive_override else raw_dir / "asm-eng.zip"
    if not archive.exists():
        _download(DATA_URL, archive)

    with zipfile.ZipFile(archive) as zipped:
        txt_name = next(name for name in zipped.namelist() if name.endswith(".txt") and not name.startswith("_"))
        lines = zipped.read(txt_name).decode("utf-8").splitlines()

    records: list[dict[str, str]] = []
    dropped = Counter()
    seen_pairs: set[tuple[str, str]] = set()
    for line in lines:
        fields = line.split("\t", maxsplit=2)
        if len(fields) != 3:
            dropped["malformed"] += 1
            continue
        source, target, attribution = map(normalize_text, fields)
        if not source or not target:
            dropped["empty"] += 1
            continue
        src_n, tgt_n = len(source.split()), len(target.split())
        if src_n > config.max_source_tokens or tgt_n > config.max_target_tokens:
            dropped["too_long"] += 1
            continue
        char_ratio = len(target) / max(1, len(source))
        if not 0.15 <= char_ratio <= 6.0:
            dropped["length_ratio"] += 1
            continue
        pair_key = (leakage_key(source), leakage_key(target))
        if pair_key in seen_pairs:
            dropped["duplicate_pair"] += 1
            continue
        seen_pairs.add(pair_key)
        records.append({"source": source, "target": target, "attribution": attribution, "source_key": pair_key[0]})

    frame = pd.DataFrame(records)
    unique_sources = sorted(frame.source_key.unique())
    rng = random.Random(config.seed)
    rng.shuffle(unique_sources)
    n_sources = len(unique_sources)
    train_end, valid_end = int(0.80 * n_sources), int(0.90 * n_sources)
    key_splits = {
        "train": set(unique_sources[:train_end]),
        "validation": set(unique_sources[train_end:valid_end]),
        "test": set(unique_sources[valid_end:]),
    }
    limits = {"train": config.max_train, "validation": config.max_validation, "test": config.max_test}
    splits: dict[str, pd.DataFrame] = {}
    for split_name, keys in key_splits.items():
        split = frame[frame.source_key.isin(keys)].sample(frac=1, random_state=config.seed).reset_index(drop=True)
        if limits[split_name] is not None:
            split = split.iloc[: limits[split_name]].copy()
        split.insert(0, "row_id", [f"{split_name}-{i:05d}" for i in range(len(split))])
        split.to_json(processed_dir / f"{split_name}.jsonl", orient="records", lines=True, force_ascii=False)
        splits[split_name] = split

    overlap = {
        "train_validation": len(set(splits["train"].source_key) & set(splits["validation"].source_key)),
        "train_test": len(set(splits["train"].source_key) & set(splits["test"].source_key)),
        "validation_test": len(set(splits["validation"].source_key) & set(splits["test"].source_key)),
    }
    manifest = {
        "dataset": "ManyThings curated Tatoeba English-Assamese sentence pairs",
        "source_url": DATA_URL,
        "source_page": DATA_PAGE,
        "license": LICENSE,
        "archive_sha256": sha256(archive),
        "seed": config.seed,
        "profile": config.profile,
        "raw_rows": len(lines),
        "clean_rows_before_caps": len(frame),
        "dropped": dict(dropped),
        "split_rows": {name: len(split) for name, split in splits.items()},
        "unique_sources": {name: int(split.source_key.nunique()) for name, split in splits.items()},
        "normalized_source_overlap": overlap,
        "target_assamese_char_fraction": {
            name: assamese_character_fraction("".join(split.target.tolist())) for name, split in splits.items()
        },
    }
    (processed_dir / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return splits, manifest


def assamese_character_fraction(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    return sum("\u0980" <= ch <= "\u09ff" for ch in letters) / max(1, len(letters))


def dataset_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in splits.items():
        rows.append(
            {
                "split": name,
                "pairs": len(frame),
                "unique_sources": frame.source_key.nunique(),
                "median_source_words": frame.source.str.split().str.len().median(),
                "median_target_words": frame.target.str.split().str.len().median(),
                "assamese_char_fraction": assamese_character_fraction("".join(frame.target)),
            }
        )
    return pd.DataFrame(rows)


def count_parameters(model: Any) -> dict[str, int]:
    # PEFT corrects logical parameter counts for packed 4-bit weights; a raw
    # ``numel`` sum otherwise under-reports QLoRA base-model parameters.
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
        return {"parameters": int(total), "trainable_parameters": int(trainable)}
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def device_name() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def clear_memory(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def train_sentencepiece(train: pd.DataFrame, root: str | Path, config: ExperimentConfig):
    import sentencepiece as spm

    output_dir = Path(root) / "artifacts" / "checkpoints" / "basic_transformer"
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "spm_corpus.txt"
    corpus_path.write_text("\n".join(train.source.tolist() + train.target.tolist()), encoding="utf-8")
    prefix = output_dir / "joint_bpe"
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(prefix),
        vocab_size=config.basic_vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        shuffle_input_sentence=False,
        num_threads=1,
        hard_vocab_limit=False,
    )
    processor = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
    return processor


def make_basic_model(vocab_size: int, config: ExperimentConfig):
    import torch
    from torch import nn

    class BasicTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.d_model = config.basic_d_model
            self.src_embedding = nn.Embedding(vocab_size, self.d_model, padding_idx=0)
            self.tgt_embedding = nn.Embedding(vocab_size, self.d_model, padding_idx=0)
            self.position = nn.Embedding(256, self.d_model)
            self.transformer = nn.Transformer(
                d_model=self.d_model,
                nhead=config.basic_heads,
                num_encoder_layers=config.basic_layers,
                num_decoder_layers=config.basic_layers,
                dim_feedforward=config.basic_ffn,
                dropout=config.basic_dropout,
                batch_first=True,
                norm_first=True,
            )
            self.output = nn.Linear(self.d_model, vocab_size)

        def _embed(self, tokens, embedding):
            positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
            return embedding(tokens) * math.sqrt(self.d_model) + self.position(positions)

        def forward(self, source, target_in):
            causal = nn.Transformer.generate_square_subsequent_mask(target_in.size(1), device=target_in.device)
            hidden = self.transformer(
                self._embed(source, self.src_embedding),
                self._embed(target_in, self.tgt_embedding),
                tgt_mask=causal,
                src_key_padding_mask=source.eq(0),
                tgt_key_padding_mask=target_in.eq(0),
                memory_key_padding_mask=source.eq(0),
            )
            return self.output(hidden)

        @torch.inference_mode()
        def translate_ids(self, source, max_new_tokens):
            self.eval()
            target = torch.full((source.size(0), 1), 2, dtype=torch.long, device=source.device)
            finished = torch.zeros(source.size(0), dtype=torch.bool, device=source.device)
            for _ in range(max_new_tokens):
                next_token = self(source, target)[:, -1].argmax(-1)
                next_token = torch.where(finished, torch.zeros_like(next_token), next_token)
                target = torch.cat([target, next_token[:, None]], dim=1)
                finished |= next_token.eq(3)
                if finished.all():
                    break
            return target[:, 1:]

    return BasicTransformer()


def _basic_loader(frame, sp, config, shuffle):
    import torch
    from torch.utils.data import DataLoader

    examples = list(zip(frame.source.tolist(), frame.target.tolist()))

    def collate(batch):
        sources, targets = zip(*batch)
        src_ids = [[2] + sp.encode(text)[: config.max_source_tokens - 2] + [3] for text in sources]
        tgt_ids = [[2] + sp.encode(text)[: config.max_target_tokens - 2] + [3] for text in targets]
        src = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in src_ids], batch_first=True, padding_value=0)
        tgt = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in tgt_ids], batch_first=True, padding_value=0)
        return src, tgt

    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(examples, batch_size=config.basic_batch_size, shuffle=shuffle, collate_fn=collate, generator=generator)


def train_basic_transformer(splits, root, config):
    import torch
    from torch import nn

    set_seed(config.seed)
    sp = train_sentencepiece(splits["train"], root, config)
    model = make_basic_model(sp.vocab_size(), config).to(device_name())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.basic_lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.basic_epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    train_loader = _basic_loader(splits["train"], sp, config, True)
    valid_loader = _basic_loader(splits["validation"], sp, config, False)
    history, best_loss, best_state, patience_left = [], float("inf"), None, 4
    start = time.perf_counter()
    for epoch in range(config.basic_epochs):
        model.train()
        train_total = 0.0
        for source, target in train_loader:
            source, target = source.to(device_name()), target.to(device_name())
            optimizer.zero_grad(set_to_none=True)
            logits = model(source, target[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), target[:, 1:].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += loss.item()
        scheduler.step()
        model.eval()
        valid_total = 0.0
        with torch.inference_mode():
            for source, target in valid_loader:
                source, target = source.to(device_name()), target.to(device_name())
                logits = model(source, target[:, :-1])
                valid_total += criterion(logits.reshape(-1, logits.size(-1)), target[:, 1:].reshape(-1)).item()
        train_loss = train_total / max(1, len(train_loader))
        valid_loss = valid_total / max(1, len(valid_loader))
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": valid_loss})
        print(f"basic epoch {epoch + 1}: train={train_loss:.4f}, validation={valid_loss:.4f}")
        if valid_loss < best_loss - 1e-4:
            best_loss = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = 4
        else:
            patience_left -= 1
            if patience_left == 0:
                break
    train_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    checkpoint = Path(root) / "artifacts" / "checkpoints" / "basic_transformer"
    torch.save({"model": best_state, "config": asdict(config)}, checkpoint / "model.pt")
    (checkpoint / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model.to(device_name()), sp, train_seconds, history


def infer_basic(model, sp, frame, config):
    import torch

    outputs, batch_size = [], config.inference_batch_size
    model.eval()
    if device_name() == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for offset in range(0, len(frame), batch_size):
        texts = frame.source.iloc[offset : offset + batch_size].tolist()
        ids = [[2] + sp.encode(x)[: config.max_source_tokens - 2] + [3] for x in texts]
        source = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in ids], batch_first=True, padding_value=0).to(device_name())
        generated = model.translate_ids(source, config.max_new_tokens).cpu().tolist()
        outputs.extend(sp.decode([[token for token in row if token not in (0, 2, 3)] for row in generated]))
    if device_name() == "cuda":
        torch.cuda.synchronize()
    return outputs, time.perf_counter() - start


def _seq2seq_collate(tokenizer, rows, config):
    import torch

    source, target = zip(*rows)
    inputs = tokenizer(
        [f"translate English to Assamese: {text}" for text in source],
        max_length=config.max_source_tokens,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    labels = tokenizer(
        text_target=list(target),
        max_length=config.max_target_tokens,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).input_ids
    labels[labels == tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return inputs


def train_mt5(splits, root, config, hf_token=None):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.mt5_model, token=hf_token)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.mt5_model, token=hf_token).to(device_name())
    rows = list(zip(splits["train"].source, splits["train"].target))
    valid_rows = list(zip(splits["validation"].source, splits["validation"].target))
    collate = lambda batch: _seq2seq_collate(tokenizer, batch, config)
    train_loader = DataLoader(rows, batch_size=config.pretrained_batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_rows, batch_size=config.pretrained_batch_size, shuffle=False, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.pretrained_lr, weight_decay=0.01)
    # mT5 is prone to fp16 overflow on T4-class GPUs. Keep the full fine-tune
    # in fp32 so the reported loss and generated translations remain valid.
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    history, best_loss, best_state = [], float("inf"), None
    start = time.perf_counter()
    for epoch in range(config.pretrained_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for step, batch in enumerate(train_loader):
            batch = {key: value.to(device_name()) for key, value in batch.items()}
            with torch.amp.autocast("cuda", enabled=False):
                loss = model(**batch).loss / config.grad_accum_steps
            scaler.scale(loss).backward()
            total += loss.item() * config.grad_accum_steps
            if (step + 1) % config.grad_accum_steps == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        model.eval()
        valid_total = 0.0
        with torch.inference_mode():
            for batch in valid_loader:
                batch = {key: value.to(device_name()) for key, value in batch.items()}
                valid_total += model(**batch).loss.item()
        train_loss, valid_loss = total / len(train_loader), valid_total / len(valid_loader)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": valid_loss})
        print(f"mT5 epoch {epoch + 1}: train={train_loss:.4f}, validation={valid_loss:.4f}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    train_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    output = Path(root) / "artifacts" / "checkpoints" / "mt5"
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model, tokenizer, train_seconds, history


def infer_seq2seq(model, tokenizer, frame, config):
    import torch

    predictions, batch_size = [], config.inference_batch_size
    model.eval()
    if device_name() == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for offset in range(0, len(frame), batch_size):
        texts = [f"translate English to Assamese: {x}" for x in frame.source.iloc[offset : offset + batch_size]]
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=config.max_source_tokens, return_tensors="pt").to(device_name())
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=config.max_new_tokens, num_beams=config.num_beams, do_sample=False)
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    if device_name() == "cuda":
        torch.cuda.synchronize()
    return predictions, time.perf_counter() - start


def translation_prompt(source: str) -> str:
    return (
        "### Instruction\nTranslate the English sentence into Assamese. Preserve names, numbers, "
        "negation and tone. Return only the Assamese translation.\n\n"
        f"### English\n{source}\n\n### Assamese\n"
    )


def _causal_collator(tokenizer, config):
    import torch

    def collate(rows):
        encoded, labels = [], []
        for source, target in rows:
            prompt_ids = tokenizer(translation_prompt(source), add_special_tokens=True, truncation=True, max_length=config.max_source_tokens).input_ids
            answer_ids = tokenizer(target + (tokenizer.eos_token or ""), add_special_tokens=False, truncation=True, max_length=config.max_target_tokens).input_ids
            ids = (prompt_ids + answer_ids)[: config.max_source_tokens + config.max_target_tokens]
            encoded.append(torch.tensor(ids, dtype=torch.long))
            labels.append(torch.tensor([-100] * len(prompt_ids) + answer_ids, dtype=torch.long)[: len(ids)])
        input_ids = torch.nn.utils.rnn.pad_sequence(encoded, batch_first=True, padding_value=tokenizer.pad_token_id)
        label_ids = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_ids": input_ids, "attention_mask": input_ids.ne(tokenizer.pad_token_id), "labels": label_ids}

    return collate


def train_causal_lora(model_id, label, splits, root, config, hf_token=None):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if device_name() != "cuda":
        raise RuntimeError(f"{label} LoRA training requires a CUDA GPU in this notebook profile.")
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_4bit = "TinyLlama" in model_id
    load_kwargs: dict[str, Any] = {
        "token": hf_token,
        "device_map": {"": 0},
        # The compact Gemma checkpoint fits comfortably in fp32 and otherwise
        # overflows on T4 fp16; TinyLlama remains NF4/fp16 for memory efficiency.
        "torch_dtype": torch.float16 if use_4bit else torch.float32,
    }
    if use_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.config.use_cache = False
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    available = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    targets = [name for name in candidates if name in available]
    if not targets:
        raise RuntimeError(f"No supported LoRA projection modules found for {model_id}")
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        ),
    )
    rows = list(zip(splits["train"].source, splits["train"].target))
    loader = DataLoader(rows, batch_size=config.pretrained_batch_size, shuffle=True, collate_fn=_causal_collator(tokenizer, config))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.pretrained_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_4bit)
    history, start = [], time.perf_counter()
    for epoch in range(config.pretrained_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for step, batch in enumerate(loader):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_4bit):
                loss = model(**batch).loss / config.grad_accum_steps
            scaler.scale(loss).backward()
            total += loss.item() * config.grad_accum_steps
            if (step + 1) % config.grad_accum_steps == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        epoch_loss = total / len(loader)
        history.append({"epoch": epoch + 1, "train_loss": epoch_loss})
        print(f"{label} LoRA epoch {epoch + 1}: train={epoch_loss:.4f}")
    train_seconds = time.perf_counter() - start
    output = Path(root) / "artifacts" / "checkpoints" / label
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    model.config.use_cache = True
    return model, tokenizer, train_seconds, history


def infer_causal(model, tokenizer, frame, config):
    import torch

    predictions = []
    model.eval()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for source in frame.source:
        inputs = tokenizer(translation_prompt(source), return_tensors="pt", truncation=True, max_length=config.max_source_tokens).to(model.device)
        prompt_length = inputs.input_ids.shape[1]
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                num_beams=config.num_beams,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        predictions.append(tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True).strip().split("###")[0].strip())
    torch.cuda.synchronize()
    return predictions, time.perf_counter() - start


def compute_metrics(references: Sequence[str], predictions: Sequence[str]) -> dict[str, float]:
    import sacrebleu

    if len(references) != len(predictions) or not references:
        raise ValueError("references and predictions must have equal, non-zero length")
    bleu = sacrebleu.corpus_bleu(list(predictions), [list(references)], tokenize="intl")
    chrf = sacrebleu.corpus_chrf(list(predictions), [list(references)], char_order=6, word_order=0)
    return {"BLEU": float(bleu.score), "chrF": float(chrf.score)}


def diagnostic_flags(source: str, prediction: str) -> list[str]:
    flags = []
    if not prediction.strip():
        flags.append("empty")
        return flags
    if assamese_character_fraction(prediction) < 0.35:
        flags.append("low_assamese_script")
    src_numbers, pred_numbers = re.findall(r"\d+(?:[.,]\d+)?", source), re.findall(r"\d+(?:[.,]\d+)?", prediction)
    if src_numbers != pred_numbers:
        flags.append("number_changed")
    words = prediction.split()
    if len(words) >= 6 and len(set(words)) / len(words) < 0.55:
        flags.append("repetition")
    ratio = len(prediction) / max(1, len(source))
    if ratio < 0.2 or ratio > 4.0:
        flags.append("length_outlier")
    if re.search(r"\b(the|is|are|was|translate|english|assamese)\b", prediction.casefold()):
        flags.append("untranslated_or_prompt_leak")
    return flags


def save_system_result(root, system, test_frame, predictions, training_seconds, inference_seconds, parameters, method):
    output_dir = Path(root) / "artifacts" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(test_frame.target.tolist(), predictions)
    rows = []
    flag_counts = Counter()
    for (_, row), prediction in zip(test_frame.iterrows(), predictions):
        flags = diagnostic_flags(row.source, prediction)
        flag_counts.update(flags)
        rows.append(
            {
                "row_id": row.row_id,
                "source": row.source,
                "reference": row.target,
                "prediction": prediction,
                "flags": flags,
            }
        )
    result = {
        "system": system,
        "method": method,
        **parameters,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "sentences_per_second": len(test_frame) / max(inference_seconds, 1e-9),
        **metrics,
        "flag_counts": dict(flag_counts),
        "test_rows": len(test_frame),
    }
    (output_dir / f"{system}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / f"{system}_translations.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return result


def save_summary(root, results: Sequence[dict[str, Any]]) -> pd.DataFrame:
    table = pd.DataFrame(results)
    columns = [
        "system", "method", "parameters", "trainable_parameters", "training_seconds",
        "inference_seconds", "sentences_per_second", "BLEU", "chrF", "test_rows",
    ]
    table = table[columns].sort_values("BLEU", ascending=False).reset_index(drop=True)
    output = Path(root) / "artifacts" / "results"
    table.to_csv(output / "summary.csv", index=False)
    (output / "summary.md").write_text(table.to_markdown(index=False), encoding="utf-8")
    return table


def environment_manifest(config: ExperimentConfig) -> dict[str, Any]:
    import torch
    import transformers

    return {
        "config": asdict(config),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
