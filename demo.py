"""Gradio demo interface for NutriScan.

Two tabs:
  1. **Fridge Analyser** — upload an image, detect produce, estimate
     portions and freshness.
  2. **Meal Planner** (offline) — compute macro deficit and score
     recipes without MongoDB.

Run with::

    uv run python demo.py
"""

from __future__ import annotations


def _analyse_fridge(image_path: str | None, user_id: str) -> str:
    """Run portion + freshness pipeline on an uploaded image."""
    if image_path is None:
        return "⚠️ Please upload an image first."

    from api.settings import settings

    ckpt = settings.freshness_checkpoint_path

    if ckpt.exists():
        from models.portion.pipeline import PortionPipeline

        pipeline = PortionPipeline(freshness_checkpoint=ckpt)
        items = pipeline.run(image_path)
    else:
        from models.portion.estimator import PortionEstimator

        estimator = PortionEstimator()
        portions = estimator.estimate(image_path)
        items = portions  # type: ignore[assignment]

    if not items:
        return "No recognised produce detected."

    header = "| Label | Grams | Uncertainty | Freshness | Freshness Label |\n"
    header += "|-------|-------|-------------|-----------|----------------|\n"
    rows: list[str] = []
    for item in items:
        if hasattr(item, "freshness_score"):
            f_score = f"{item.freshness_score:.2f}"
            f_label = getattr(item, "freshness_label", "unknown")
        else:
            f_score = "—"
            f_label = "N/A (no checkpoint)"
        rows.append(
            f"| {item.label} "
            f"| {item.estimated_grams:.1f}g "
            f"| ±{item.uncertainty_grams:.1f}g "
            f"| {f_score} "
            f"| {f_label} |"
        )
    note = ""
    if not ckpt.exists():
        note = (
            "\n\n> ⚠️ Freshness model not trained yet. "
            "Run: `uv run python -m models.freshness.train`"
        )
    return f"**User:** {user_id}\n\n{header}" + "\n".join(rows) + note


def _plan_meal(user_id: str, meal_type: str) -> str:
    """Offline meal planning — no MongoDB required."""
    from datetime import UTC, datetime

    from agent.tools import compute_macro_deficit, load_recipes, score_recipe
    from db.models import (
        BoundingBox,
        DetectedItem,
        FridgeState,
        MacroTargets,
        UserProfile,
    )

    profile = UserProfile(
        user_id=user_id,
        display_name="Demo User",
        daily_targets=MacroTargets(calories=2200, protein_g=60, carbs_g=275, fat_g=73),
        dietary_restrictions=["vegetarian"],
    )
    fridge = FridgeState(
        user_id=user_id,
        image_path="demo",
        captured_at=datetime.now(tz=UTC),
        detected_items=[
            DetectedItem(
                name="apple",
                bounding_box=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100),
                confidence=0.9,
                freshness_score=0.85,
                estimated_grams=180.0,
            ),
            DetectedItem(
                name="banana",
                bounding_box=BoundingBox(x_min=0, y_min=0, x_max=80, y_max=120),
                confidence=0.88,
                freshness_score=0.7,
                estimated_grams=120.0,
            ),
        ],
    )

    deficit = compute_macro_deficit(profile.daily_targets, [])

    try:
        recipes = load_recipes()
    except FileNotFoundError:
        return "❌ `data/recipes.json` not found."

    scored = []
    for r in recipes:
        s = score_recipe(r, deficit, fridge)
        scored.append((r, s))
    scored.sort(key=lambda x: x[1], reverse=True)

    header = "| Rank | Recipe | Score | Calories | Protein |\n"
    header += "|------|--------|-------|----------|---------|\n"
    rows = []
    for i, (r, s) in enumerate(scored[:3], 1):
        m = r.get("macros", {})
        rows.append(
            f"| {i} | {r['name']} | {s:.3f} "
            f"| {m.get('calories', 0)} kcal "
            f"| {m.get('protein_g', 0)}g |"
        )
    return (
        f"**User:** {user_id} · **Meal:** {meal_type}\n\n"
        f"**Macro deficit:** {deficit.calories:.0f} kcal / "
        f"{deficit.protein_g:.0f}g P / {deficit.carbs_g:.0f}g C / "
        f"{deficit.fat_g:.0f}g F\n\n" + header + "\n".join(rows)
    )


def main() -> None:
    """Launch the Gradio demo."""
    import gradio as gr

    with gr.Blocks(title="NutriScan Demo") as demo:
        gr.Markdown("# 🥗 NutriScan — Demo")

        with gr.Tab("Fridge Analyser"):
            img = gr.Image(type="filepath", label="Fridge / produce photo")
            uid = gr.Textbox(value="demo_user", label="User ID")
            btn = gr.Button("Analyse", variant="primary")
            out = gr.Markdown(label="Results")
            btn.click(_analyse_fridge, inputs=[img, uid], outputs=out)

        with gr.Tab("Meal Planner (offline — no MongoDB required)"):
            uid2 = gr.Textbox(value="demo_user", label="User ID")
            mt = gr.Dropdown(
                choices=["breakfast", "lunch", "dinner", "snack"],
                value="dinner",
                label="Meal type",
            )
            btn2 = gr.Button("Plan", variant="primary")
            out2 = gr.Markdown(label="Meal Plan")
            btn2.click(_plan_meal, inputs=[uid2, mt], outputs=out2)

    demo.launch()


if __name__ == "__main__":
    main()
