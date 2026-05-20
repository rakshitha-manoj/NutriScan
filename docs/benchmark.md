# Benchmark Report — NutriScan

## Freshness Model

| Property | Value |
|----------|-------|
| Dataset | Fruits Fresh-and-Rotten (Kaggle) |
| Categories | 6 (apple, banana, orange — fresh & rotten) |
| Split (seed 42) | 80 / 10 / 10 (train / val / test) |
| Architecture | CLIP ViT-B/32 (frozen) → MLP regression head |
| MLP layers | Linear(512,256) → BN → ReLU → Dropout(0.3) → Linear(256,64) → ReLU → Dropout(0.2) → Linear(64,1) → Sigmoid |
| Loss | MSE |
| Optimiser | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=50) |
| Early stopping | patience 10 on val loss |

### Results

| Metric | Value |
|--------|-------|
| Test MAE | 0.0025 |
| Test RMSE | 0.0348 |
| Accuracy @ 0.5 | 99.85% |
| Naive baseline MAE (predict 0.5) | 0.5000 |
| Improvement over baseline | 99.5% |

> **Note on uncertainty:** MC Dropout (20 forward passes with dropout
> enabled) provides an uncertainty estimate. This is most meaningful
> near the 0.5 decision boundary, where the model is least confident
> about the fresh/rotten classification.

> **Why the metrics are so strong:** Labels are binary (1.0 = fresh,
> 0.0 = rotten), so the task is effectively binary classification
> reframed as regression. The strong metrics reflect the separability
> of CLIP ViT-B/32 embeddings on this distribution, not a hard
> regression problem. Uncertainty estimates from MC Dropout are
> meaningful primarily near the decision boundary.

## Portion Estimation

| Property | Value |
|----------|-------|
| Detector | YOLOv8n (pretrained COCO, used as-is) |
| Method | Geometric depth proxy: pixel area → cm² → cm³ → grams |
| Reference calibration | Dinner plate (26 cm ⌀, ~531 cm²) |
| Fallback | Image-area heuristic when no reference detected |
| Uncertainty source | Bounding box aspect ratio deviation from expected |
| Food categories | 10 (apple, banana, orange, carrot, broccoli, tomato, potato, cucumber, grape, egg) |

### Limitations

- No ground-truth gram weights are available in the dataset.
- The geometric approximation is a valid first-order approach but
  cannot account for occlusion, stacking, or density variation.
- Estimates are best interpreted as order-of-magnitude guidance.

## Agent

| Property | Value |
|----------|-------|
| Framework | LangGraph StateGraph (deterministic, no LLM) |
| Nodes | 7 (see below) |
| Recipe corpus | 30 recipes covering breakfast, lunch, dinner, snack |

### Graph nodes

1. **load_profile** — Validate user profile is present
2. **load_fridge** — Validate fridge state is present
3. **compute_deficit** — Remaining macro budget from today's logs
4. **filter_recipes** — Filter by dietary restrictions + fridge overlap
5. **score_recipes** — Score each candidate against macro deficit
6. **select_plan** — Pick top 3 recipes, compute projected macros
7. **persist_log** — Mark state for DB persistence

### Scoring formula

```
score = 0.7 × macro_fit + 0.3 × freshness_bonus
```

- **macro_fit** = `1 − mean(|recipe_macro − deficit_macro| / max(deficit_macro, 1))`  
  Clamped to [0, 1]. Measures how well a recipe fills the remaining budget.
- **freshness_bonus** = mean freshness score of fridge items that the
  recipe uses. 0 if no fridge overlap.

## Reproducibility

```bash
# From a clean clone (seed=42 everywhere):
uv sync --all-extras --dev
uv run python -m data.download
uv run python -m models.freshness.preprocess
uv run python -m models.freshness.train
```
