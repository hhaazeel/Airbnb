# Edinburgh Airbnb Performance Prediction with LLM-Derived Spatial Semantics

This repository contains the code and analysis for an MSc Urban Analytics group
project investigating how **LLM-derived spatial semantics** and **point-of-interest
(POI) density** predict the performance of Edinburgh Airbnb listings.

## Overview

We extend Wang et al.'s framework by predicting **two complementary outcomes**:

- **`log_price`** — what hosts charge (market valuation)
- **`log_review_count`** — what guests actually engage with (demand signal)

Listing descriptions from October 2024 (~5,100 listings) are processed through
four LLM variants (Qwen3 description-only / combined, Nemo description-only /
combined) to extract three categories of place references:

- `specific_locations` — named, mappable places (e.g. "Edinburgh Castle")
- `general_references` — vague nearby references (e.g. "shops", "city centre")
- `parent_references` — the listing's own neighbourhood (excluded from analysis)

The number of unique `specific + general` references becomes our core variable
**`spatial_semantics`**, which we combine with POI density at three radii (500m,
800m, 1500m) and a comprehensive set of structural, host, policy, sentiment,
and amenity controls.

## Pipeline

1. **Data preparation** — listing cleaning, host gender inference, POI buffer
   counts, review-level VADER/DistilBERT sentiment analysis.
2. **LLM inference & matching** — process LLM JSON outputs, deduplicate, align
   to the October 2024 snapshot via a two-step matching strategy (id match +
   description text match), and compute `spatial_semantics`.
3. **Machine learning** — 20 experiments per target (M_base + 4 LLM-only +
   3 POI-only + 12 LLM × POI combined) across 11 algorithms (Lasso, Ridge,
   Elastic Net, Kernel Ridge, Bayesian Ridge, Random Forest, Gradient Boost,
   XGBoost ×2, LightGBM, Averaged Ensemble) with Hyperopt-tuned XGBoost
   (300 trials) and LightGBM (200 trials).
4. **Cross-dataset validation** — the same pipeline replicated on three
   Edinburgh snapshots (Feb / Aug / Oct 2024) to test temporal robustness.
5. **Validation & residuals** — manual annotation of LLM outputs and residual diagnostics on the best models.
