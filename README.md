# Assignment 4 — Low-resource English → Assamese Machine Translation

## Basic Transformer, mT5, TinyLlama, and Gemma

**Student Name:** Pavit Agrawal  
**Registration Number:** 230953406  
**Course:** Foundations of Generative AI (ICT4403)  
**Language Pair:** English (source) → Assamese (target)  
**Dataset:** ManyThings curated Tatoeba English–Assamese parallel corpus  
**Reported Run:** `quick` profile on an NVIDIA Tesla T4 (Google Colab)  
**Environment:** Python 3.13.16, PyTorch 2.11.0+cu130, Transformers 4.57.6, CUDA 13.0

This repository contains the executed Jupyter notebook, the exact train/validation/test splits used, the generated translations for every system, the metric files, plots, and the full run manifest. All numbers in this README come from the executed notebook; nothing is hard-coded or estimated.

---

## 1. Problem Statement and Objective

The objective is to build and compare four machine-translation systems for the low-resource English → Assamese direction on one fixed, leakage-safe test set:

1. **Basic Transformer** — encoder–decoder trained from scratch.
2. **mT5-small** — pretrained multilingual encoder–decoder, fully fine-tuned.
3. **TinyLlama 1.1B Chat** — decoder-only LLaMA-architecture model, adapted with NF4 QLoRA.
4. **Gemma 3 270M IT** — compact instruction-tuned decoder-only model, adapted with LoRA.

Every system is evaluated with:

- **BLEU** and **chrF** (corpus-level, SacreBLEU),
- training/adaptation time and inference time,
- throughput in sentences per second,
- total and trainable parameter counts,
- a qualitative and responsible-AI review of the generated translations.

The comparison also discusses encoder–decoder versus decoder-only trade-offs while explicitly accounting for confounders such as model size, pretraining data, tokenizer, quantization, and adaptation method.

---

## 2. Dataset Description and Public Source Link

- **Dataset:** ManyThings curated Tatoeba English–Assamese sentence pairs
- **Source page:** <https://www.manythings.org/anki/>
- **Source archive:** <https://www.manythings.org/anki/asm-eng.zip>
- **Original project:** [Tatoeba](https://tatoeba.org/)
- **License:** CC BY 2.0 FR (Tatoeba; per-sentence attribution is retained)
- **Archive SHA-256:** `7fb72adccf0e7c5cbac9e83554ae0929924e0d1d317fb693e8508992c4e090ae`

### Preprocessing and split

- Unicode NFC normalization, whitespace/control cleanup, empty and malformed row removal.
- Conservative length filters (`max_source_tokens = 64`, `max_target_tokens = 64`).
- Character-length-ratio filter and exact normalized-pair deduplication.
- Deterministic grouped split by normalized English source, so alternative translations of the same source cannot leak across splits.
- Seed `42`, 80/10/10 grouped split before profile caps.
- The executed `quick` run caps the splits at **1,000 train / 100 validation / 100 test** pairs.

| Split | Pairs | Unique sources | Median source words | Median target words | Assamese character fraction |
|---|---:|---:|---:|---:|---:|
| Train | 1,000 | 945 | 5.0 | 5.0 | 1.000000 |
| Validation | 100 | 97 | 5.0 | 4.0 | 0.998476 |
| Test | 100 | 93 | 5.0 | 5.0 | 1.000000 |

Source-overlap audit (normalized English source): **0** for train/validation, train/test, and validation/test.

The raw archive contained 7,278 rows; 7,264 rows remained after cleaning and 14 duplicate pairs were removed. The committed JSONL files in [`data/`](data/) are exactly the splits used by the executed run and are reproduced deterministically by the notebook.

---

## 3. Models Implemented

| System | Architecture | Adaptation | Total parameters | Trainable parameters | Tokenizer |
|---|---|---|---:|---:|---|
| **Basic Transformer** | Encoder–decoder | Trained from scratch | 9,746,360 | 9,746,360 | SentencePiece BPE (3,000 vocab, fit on train split) |
| **mT5-small** | Encoder–decoder | Full fine-tuning | 300,176,768 | 300,176,768 | Multilingual SentencePiece (250k vocab) |
| **TinyLlama 1.1B Chat** | Decoder-only (LLaMA architecture) | NF4 QLoRA | 1,106,356,224 | 6,307,840 | Byte-level BPE |
| **Gemma 3 270M IT** | Decoder-only | LoRA | 269,996,672 | 1,898,496 | SentencePiece with byte fallback |

### Architecture and training details

- **Basic Transformer:** 4 encoder and decoder layers, `d_model = 256`, 8 attention heads, feed-forward width 1,024, dropout 0.1, label smoothing, AdamW (`lr = 3e-4`), cosine decay, gradient clipping, validation-loss checkpointing, and early stopping. Trained for 5 epochs.
- **mT5-small:** Fully fine-tuned for 1 epoch with the task prefix `translate English to Assamese:`, batch size 4, gradient accumulation 4, and learning rate `5e-5`.
- **TinyLlama 1.1B Chat:** Loaded in NF4 4-bit form and adapted with LoRA (`r = 8`, `alpha = 16`, dropout 0.05) on attention/MLP projections, with prompt-masked loss so only the Assamese answer is learned. One adaptation epoch. TinyLlama is an independent LLaMA-architecture checkpoint, not an official Meta LLaMA release.
- **Gemma 3 270M IT:** Adapted with the same LoRA configuration, prompt format, epoch count, and greedy decoding policy as TinyLlama. The Gemma license must be accepted on Hugging Face before running.

All systems receive identical source strings, use greedy decoding (`num_beams = 1`, no sampling), batch size 1, and a maximum of 48 new tokens. Training time excludes model download; inference time excludes model loading and covers the full 100-sentence test set.

---

## 4. Execution Steps and Dependencies

### Environment

The reported measurements were produced on an **NVIDIA Tesla T4 (16 GB)** in Google Colab. The full run manifest, including library versions and hardware, is in [`results/run_manifest.json`](results/run_manifest.json).

### Local / Colab setup

```bash
git clone https://github.com/pavitagrawal/FGAI_IA4.git
cd FGAI_IA4
pip install -r requirements.txt
```

Then open **`FGAI_Assignment4.ipynb`** and run all cells from top to bottom.

For Gemma:

1. Accept the `google/gemma-3-270m-it` license on Hugging Face.
2. Provide `HF_TOKEN` as a Colab secret or environment variable.

The data archive is bundled as `asm-eng.zip`, so no download is required. The notebook also downloads it automatically if the archive is missing.

### Reproducible run order

```text
1. Install dependencies and resolve the project root.
2. Prepare and persist the train/validation/test splits.
3. Train the Basic Transformer from scratch and decode the test set.
4. Fine-tune mT5-small and decode the test set.
5. Adapt TinyLlama with NF4 QLoRA and decode the test set.
6. Adapt Gemma 3 270M IT with LoRA and decode the test set.
7. Compute BLEU/chrF, timings, throughput, and parameter counts.
8. Run the diagnostic audit and write the review sheet.
9. Write the run manifest and the automatically generated conclusion.
```

Optional checks:

```bash
pytest -q
```

The reusable, tested implementation lives in [`src/mt_assignment.py`](src/mt_assignment.py); the notebook is the experiment entry point.

> The notebook uses the `quick` profile (`max_train=1000`, `max_validation=100`, `max_test=100`, 5 scratch epochs, 1 pretrained epoch). This keeps the full four-model comparison feasible on a free T4. The profile limits are defined in the notebook configuration cell.

---

## 5. Results Obtained

Corpus-level **BLEU** and **chrF** are reported on the same 100 test sentences. SacreBLEU scores are on a 0–100 scale; values are low because the quick profile trains the pretrained systems for only one epoch and the scratch model for five.

| System | Architecture | Adaptation | BLEU | chrF | Training (s) | Inference (s) | Throughput (sent/s) | Total params | Trainable params |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **TinyLlama 1.1B Chat** | Decoder-only | NF4 QLoRA | **0.6083** | **9.1780** | 107.997 | 287.838 | 0.347 | 1,106,356,224 | 6,307,840 |
| **Gemma 3 270M IT** | Decoder-only | LoRA | 0.3132 | 7.6139 | 49.119 | 89.530 | 1.117 | 269,996,672 | 1,898,496 |
| **mT5-small** | Encoder–decoder | Full fine-tuning | 0.0634 | 0.0080 | 36.671 | 4.832 | 20.696 | 300,176,768 | 300,176,768 |
| **Basic Transformer** | Encoder–decoder | From scratch | 0.0000 | 4.1662 | **5.477** | **2.189** | **45.681** | 9,746,360 | 9,746,360 |

All systems produced exactly one translation for every one of the 100 shared test rows. The final integrity gate in the notebook passes.

### Plots and metric files

- [`results/metric_comparison.png`](results/metric_comparison.png) — BLEU bars and quality-versus-parameter-count view.
- [`results/eda_length_distribution.png`](results/eda_length_distribution.png) — English and Assamese length distributions.
- [`results/training_loss.png`](results/training_loss.png) — recorded training/validation loss curves.
- [`results/summary.csv`](results/summary.csv) — machine-readable metric table.
- [`results/summary.md`](results/summary.md) — Markdown metric table.
- [`results/dataset_summary.csv`](results/dataset_summary.csv) — split statistics.
- [`results/training_history.csv`](results/training_history.csv) — epoch-level loss values.
- [`results/human_review_sheet.csv`](results/human_review_sheet.csv) — flagged test rows for human review.
- [`results/run_manifest.json`](results/run_manifest.json) — complete configuration and environment manifest.
- [`results/conclusion.md`](results/conclusion.md) — automatically generated conclusion.
- [`outputs/`](outputs/) — full test-set translations for each system.

---

## 6. Model Comparison and Evaluation

### Automatic quality

- **TinyLlama 1.1B Chat with NF4 QLoRA** obtained the highest automatic scores in this run: **0.6083 BLEU / 9.1780 chrF**. Its large multilingual/instruction pretraining helps even with one adaptation epoch, but it is also the slowest system at **0.347 sentences/s** (287.8 s for 100 test sentences).
- **Gemma 3 270M IT with LoRA** was second at **0.3132 BLEU / 7.6139 chrF** and was roughly three times faster than TinyLlama during inference (**1.117 sentences/s**). Its 270M parameter count is much smaller than TinyLlama's 1.1B.
- **mT5-small** reached only **0.0634 BLEU / 0.0080 chrF**. With a single fine-tuning epoch it frequently emitted span-corruption sentinel tokens such as `<extra_id_0>` instead of fluent Assamese, which explains the near-zero chrF.
- **The Basic Transformer** reached **0.0000 BLEU / 4.1662 chrF**. Trained from random weights on 1,000 pairs, it collapsed toward a frequent token (`টমে।`, “Tom.”) and produced many length-outlier outputs.

### Computational requirements

- The **Basic Transformer** is by far the cheapest: 5.5 s to train, 2.2 s to decode the test set, and 45.7 sentences/s. It is also the smallest model at 9.7M parameters.
- **mT5-small** has the fastest encoder–decoder inference among the pretrained systems (**20.7 sentences/s**) and the lowest pretrained training time (36.7 s), but full fine-tuning updates all ~300M weights.
- **Gemma** updates only ~1.9M LoRA parameters and is a good quality/compute compromise in this quick run.
- **TinyLlama** updates ~6.3M LoRA parameters but its 4-bit base model still dominates memory and latency; inference is token-by-token and therefore slow on a T4.

### Encoder–decoder versus decoder-only

The decoder-only systems outperformed the encoder–decoder systems on automatic quality in this run (mean decoder-only BLEU 0.46 versus mean encoder–decoder BLEU 0.03). This is **descriptive, not causal**. The systems differ in:

- pretraining corpus and objective,
- tokenizer and vocabulary coverage of Assamese,
- parameter count and quantization,
- adaptation method (full fine-tuning, QLoRA, LoRA, or training from scratch),
- number of optimization steps.

The scratch Transformer isolates the difficulty of learning translation from this small corpus. mT5 adds multilingual pretraining but needs more than one epoch to move away from its span-corruption behavior. TinyLlama and Gemma bring instruction-following and broad multilingual pretraining, which is a large advantage in a low-data setting. A controlled architecture ablation would require matching data, tokenizer, parameter budget, and optimization; this assignment does not claim one.

---

## 7. Qualitative and Responsible-AI Analysis

The notebook runs heuristic diagnostics rather than automatic “correctness” labels. It flags empty output, low Assamese-script fraction, changed numbers, repetition, extreme length, and obvious English/prompt leakage. Flag counts from the executed run:

| System | Flagged rows | Main signals |
|---|---:|---|
| Basic Transformer | 71 | `length_outlier` (71) |
| mT5-small | 100 | `low_assamese_script` (100), `number_changed` (100), `length_outlier` (2) |
| TinyLlama LoRA | 3 | `empty` (1), `repetition` (2) |
| Gemma LoRA | 15 | `repetition` (14), `number_changed` (1) |

Representative failure modes from [`results/human_review_sheet.csv`](results/human_review_sheet.csv):

- **Basic Transformer** — source: *“Do you know what to bring?”* → prediction: `টমে।` (“Tom.”). The model collapses to a frequent token and ignores the source.
- **mT5-small** — source: *“Will you take me there?”* → prediction: `<extra_id_0>`. The span-corruption sentinel leaks into the output, a direct consequence of insufficient fine-tuning.
- **Gemma LoRA** — source: *“I have no problem with Tom.”* → prediction dominated by repeated `তোমই` (“you”), flagged as repetition.
- **TinyLlama LoRA** — only three flagged rows, but the largest model is also the slowest; automatic quality does not remove the need for human review.

### Responsible and ethical AI concerns

1. **Bias and representation.** Tatoeba is volunteer-contributed short text and is not representative of Assamese dialects, registers, domains, or communities. Gender, honorifics, kinship terms, caste/religion references, geography, and code-switching can be distorted. Provenance and limitations must be reported, and evaluation should include community-curated dialect and register sets with compensated native-speaker review.
2. **Hallucination and omission.** Decoder-only models may continue the prompt or add plausible facts; all systems may drop negation, names, units, or clauses. Mitigations include deterministic decoding, terminology constraints, number/name consistency checks, confidence or quality estimation, and human approval for consequential text.
3. **Cultural misrepresentation.** Kinship, honorifics, idioms, religious terms, and culturally specific names may be normalized into majority-language assumptions. Maintain community-reviewed glossaries, preserve ambiguous wording, show the source beside the translation, and allow correction/appeal.
4. **Sensitive and harmful content.** A translator should preserve meaning without intensifying slurs, threats, self-harm, sexual content, or political claims. Do not silently censor; flag sensitive passages, restrict downstream automation, protect reviewer wellbeing, and use trained bilingual reviewers with clear escalation rules.
5. **Privacy, consent, and licensing.** Retain Tatoeba attribution and license notices. Do not fine-tune on private messages or identifiable data without a lawful basis and consent. Minimize logs, encrypt stored text, define retention limits, and document model and dataset revisions.
6. **Deployment controls.** Publish a model card with subgroup/domain tests, known failure modes, intended and non-intended uses, hardware/energy measurements, and rollback/versioning. Monitor script mismatch, repetition, changed entities/numbers, and user complaints after deployment.

These systems are research prototypes. They must not be used as the sole translator for medical, legal, emergency, immigration, or public-service content.

---

## 8. Sampling-Temperature Analysis

The submitted run used **deterministic greedy decoding** (`num_beams = 1`, no sampling) for all four systems. This was chosen so that every system is evaluated under the same decoding policy and so that inference-time measurements are comparable. Because no stochastic decoding was performed, **no temperature sweep is reported and no temperature numbers are claimed**.

The notebook and `src/mt_assignment.py` can be extended to sweep `T ∈ {0.0, 0.3, 0.7, 1.0}` for the decoder-only systems. The existing qualitative evidence already shows the relevant failure modes that a temperature sweep would quantify:

- low-temperature/greedy decoding is more literal and repetitive for under-trained models,
- higher temperatures would be expected to increase lexical diversity but also synonym drift, hallucinated preambles, and number/name changes,
- the audit flags in `results/human_review_sheet.csv` provide a template for measuring those changes.

Reporting a temperature table without running it would be fabrication, so this repository reports only what the executed notebook actually measured.

---

## 9. Conclusion: Best-Performing Model

On the fixed 100-sentence test set, **TinyLlama 1.1B Chat with NF4 QLoRA is the best-performing system by automatic quality**, reaching **0.6083 BLEU and 9.1780 chrF**. Gemma 3 270M IT with LoRA is second (**0.3132 BLEU / 7.6139 chrF**) and is substantially faster. The encoder–decoder systems did not perform well in this quick profile: mT5-small leaked `<extra_id_0>` sentinels after one epoch, and the scratch Transformer collapsed to repetitive high-frequency output.

The result should be read as a low-resource, quick-profile comparison rather than a production benchmark. TinyLlama wins on quality but is the most expensive system to run here; the Basic Transformer is the cheapest and fastest but is not usable as a translator without much more data and training; mT5 and Gemma would likely improve with additional fine-tuning epochs and a larger training split. The ranking is also confounded by pretraining, tokenization, model size, quantization, and adaptation method, so it should not be interpreted as a clean causal test of encoder–decoder versus decoder-only architectures.

---

## 10. Repository Layout

```text
FGAI_Assignment4.ipynb          # executed notebook with outputs (student details included)
README.md                       # this file
requirements.txt                # Python dependencies
asm-eng.zip                     # public Tatoeba English–Assamese archive
data/
  train.jsonl                   # exact 1,000-pair training split used
  validation.jsonl              # exact 100-pair validation split used
  test.jsonl                    # exact 100-pair test split used
  dataset_manifest.json         # counts, hashes, overlap audit, script fractions
outputs/
  basic_transformer_translations.jsonl
  mt5_small_translations.jsonl
  tinyllama_lora_translations.jsonl
  gemma_lora_translations.jsonl
results/
  summary.csv                   # BLEU, chrF, timings, throughput, parameters
  summary.md
  basic_transformer.json
  mt5_small.json
  tinyllama_lora.json
  gemma_lora.json
  human_review_sheet.csv        # flagged rows for qualitative review
  dataset_summary.csv
  training_history.csv
  training_loss.png
  metric_comparison.png
  eda_length_distribution.png
  run_manifest.json             # full environment + configuration + results manifest
  conclusion.md                 # automatically generated conclusion
src/
  mt_assignment.py              # tested experiment utilities
  __init__.py
tests/
  test_pipeline.py              # unit tests and invariants
```

---

## 11. Attribution and References

- Tatoeba Project. *Tatoeba: Collection of sentences and translations*, CC BY 2.0 FR. <https://tatoeba.org/>
- ManyThings. *Tab-delimited Bilingual Sentence Pairs from Tatoeba*. <https://www.manythings.org/anki/>
- Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS.
- Xue et al. (2021). *mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer*. NAACL.
- Zhang et al. (2024). *TinyLlama: An Open-Source Small Language Model*. arXiv:2401.02385.
- Gemma Team (2025). *Gemma 3 Technical Report*. arXiv:2503.19786.
- Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR.
- Post (2018). *A Call for Clarity in Reporting BLEU Scores*. WMT.
- Popović (2015). *chrF: character n-gram F-score for automatic MT evaluation*. WMT.

Per-sentence Tatoeba attribution is preserved in the `attribution` field of every JSONL split.
