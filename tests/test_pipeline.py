import json
import zipfile
from pathlib import Path

import pandas as pd

from src.mt_assignment import (
    ExperimentConfig,
    assamese_character_fraction,
    compute_metrics,
    diagnostic_flags,
    leakage_key,
    normalize_text,
    prepare_dataset,
    translation_prompt,
)


def test_normalization_and_leakage_key():
    assert normalize_text("  Hello\t  world  ") == "Hello world"
    assert leakage_key("Hello, WORLD!") == leakage_key("hello world")


def test_assamese_script_and_flags():
    assert assamese_character_fraction("এইটো অসমীয়া বাক্য।") > 0.95
    flags = diagnostic_flags("I have 2 books.", "মোৰ 3 খন কিতাপ আছে।")
    assert "number_changed" in flags
    assert "low_assamese_script" not in flags


def test_metric_identity_is_perfect():
    references = [
        "মই আজি ৰাতিপুৱা বজাৰলৈ যাব বিচাৰোঁ।",
        "এইখন এখন অতি সুন্দৰ আৰু পুৰণি কিতাপ।",
    ]
    metrics = compute_metrics(references, references)
    assert metrics["BLEU"] > 99.9
    assert metrics["chrF"] > 99.9


def test_prompt_requests_only_translation():
    prompt = translation_prompt("Good morning.")
    assert "Good morning." in prompt
    assert prompt.endswith("### Assamese\n")
    assert "Return only" in prompt


def test_group_split_has_no_source_leakage(tmp_path: Path):
    rows = []
    for i in range(50):
        # Include two translations for some identical sources to exercise grouping.
        rows.append(f"English sentence {i}.\tঅসমীয়া বাক্য {i}।\tCC-BY attribution {i}")
        if i % 10 == 0:
            rows.append(f"English sentence {i}.\tবিকল্প বাক্য {i}।\tCC-BY attribution alt {i}")
    archive = tmp_path / "tiny.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("asm.txt", "\n".join(rows))
    config = ExperimentConfig(max_train=None, max_validation=100, max_test=100)
    splits, manifest = prepare_dataset(tmp_path, config, archive)
    assert all(len(frame) > 0 for frame in splits.values())
    assert not any(manifest["normalized_source_overlap"].values())
    keys = {name: set(frame.source_key) for name, frame in splits.items()}
    assert keys["train"].isdisjoint(keys["validation"])
    assert keys["train"].isdisjoint(keys["test"])


def test_notebook_is_valid_json():
    notebook = Path(__file__).parents[1] / "notebooks" / "English_Assamese_MT_Comparison.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    assert len(payload["cells"]) >= 20
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    for required in ("train_basic_transformer", "train_mt5", "train_causal_lora", "save_system_result"):
        assert required in text
