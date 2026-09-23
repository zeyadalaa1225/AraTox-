# AraTox — Arabic Multi-Label Toxicity Classification

A multi-label text classification system for detecting toxic and hateful language in Arabic text. The model assigns one or more of seven labels to a given piece of text: **Cussing, Hatred, Appearance, Racial, Sexual, Violence**, and **NOT** (not toxic).

This repository documents an end-to-end comparison across five modeling approaches, from classical machine learning baselines to a fine-tuned Arabic transformer (AraBERT).

## Motivation

Arabic NLP presents unique challenges — dialectal variation, rich morphology, inconsistent orthography (letter variants, diacritics, elongation characters), and comparatively limited high-quality labeled resources for toxicity detection. This project explores how far classical methods can go on this task, and how much a modern pretrained transformer improves on them, while keeping the modeling choices explainable and reproducible.

## Pipeline Overview

1. **Preprocessing** — Unicode normalization, removal of non-Arabic (Latin) characters, normalization of Arabic letter variants (e.g. أ/إ/آ → ا, ة → ه where appropriate), and stripping of diacritics and tatweel (kashida).
2. **Modeling** — Five approaches were trained and evaluated on the same train/validation split for a fair comparison.
3. **Evaluation** — Micro and Macro F1 were used as the primary metrics, since the label distribution is imbalanced (e.g. "Cussing" and "NOT" are far more frequent than "Violence" or "Racial"). Macro F1 is the headline metric, as it weighs all classes equally regardless of frequency.
4. **Threshold tuning** — For the transformer and neural models, per-label decision thresholds were tuned on validation data (rather than using a fixed 0.5 cutoff) to better handle class imbalance in a multi-label setting.

## Models Explored

| # | Approach | Description |
|---|----------|-------------|
| 1 | TF-IDF + word n-grams | Classical bag-of-words baseline with linear classifiers |
| 2 | TF-IDF + character n-grams | Character-level features, more robust to spelling/dialect variation |
| 3 | BiLSTM + Attention (trainable embeddings) | Embeddings learned from scratch, jointly with the classifier |
| 4 | BiLSTM + Attention (pretrained embeddings) | Same architecture, initialized with pretrained Arabic word embeddings |
| 5 | AraBERT (`aubmindlab/bert-base-arabertv02`) | Transformer fine-tuning, with the last 2 encoder layers unfrozen and the rest frozen for efficient training, mixed-precision (AMP), gradient clipping, a linear warmup schedule, and early stopping on validation Macro F1 |

## Results

| Model | Micro F1 | Macro F1 |
|-------|----------|----------|
| TF-IDF (word n-grams) | 0.91 | 0.8813 |
| TF-IDF (character n-grams) | 0.91 | 0.8813 |
| BiLSTM + Attention (trainable embeddings) | 0.9028 | 0.8820 |
| BiLSTM + Attention (pretrained embeddings) | 0.8850 | 0.8663 |
| **AraBERT (fine-tuned)** | **0.9022** | **0.8878** |

**AraBERT achieved the best Macro F1 (0.8878)**, confirming that a pretrained Arabic transformer generalizes better across all seven classes — including the harder, lower-support ones like *Racial* and *Violence* — than either the classical or the from-scratch deep learning baselines.

Interestingly, the classical TF-IDF baselines and the trainable-embedding BiLSTM were both very competitive, and the BiLSTM with *pretrained* embeddings underperformed its from-scratch counterpart — likely due to a mismatch between the embedding vocabulary/domain and the informal, dialectal nature of the dataset, and/or out-of-vocabulary tokens at inference time.

### Per-class takeaways (from the classical baseline's classification report, representative of the general difficulty pattern across all models)
- **Sexual, Cussing, NOT** — highest precision and recall; these classes have the clearest lexical signals and/or the most support.
- **Racial, Violence** — the hardest classes, with recall consistently trailing precision, reflecting fewer training examples and more contextual/implicit signals that are harder for surface-level features to capture.
