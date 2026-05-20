# 🥗 NutriScan — Agentic Nutrition Planner

An end-to-end AI system that scans fridge photos, estimates food freshness and portions, and plans meals using a stateful LangGraph agent.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Fridge      │────▶│  YOLOv8n     │────▶│  CLIP ViT-B/32  │
│  Photo       │     │  Detection   │     │  Feature Extr.  │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                    ┌──────────────┐     ┌─────────▼────────┐
                    │  Portion     │◀────│  Freshness MLP   │
                    │  Estimator   │     │  Regression Head │
                    └──────┬───────┘     └──────────────────┘
                           │
                    ┌──────▼───────┐     ┌─────────────────┐
                    │  LangGraph   │────▶│  Recipe Scoring  │
                    │  Agent       │     │  & Meal Plan     │
                    └──────┬───────┘     └─────────────────┘
                           │
                    ┌──────▼───────┐
                    │  FastAPI     │
                    │  + MongoDB   │
                    └──────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Dependency Manager | [uv](https://docs.astral.sh/uv/) |
| CV Backbone | CLIP ViT-B/32 (frozen, via `open-clip-torch`) |
| Regression Head | MLP on CLIP embeddings |
| Object Detection | YOLOv8n (`ultralytics`) |
| Agentic Framework | LangGraph + langchain-core |
| API Layer | FastAPI with async PyMongo |
| Database | MongoDB 7 (Docker for dev) |
| Containerisation | Docker + docker-compose |
| Testing | pytest + httpx |
| Linting | ruff + mypy |

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Docker Desktop (for MongoDB)

### Setup

```bash
# Clone the repository
git clone https://github.com/rakshitha-manoj/NutriScan.git
cd NutriScan

# Install all dependencies (including dev)
uv sync --all-extras --dev

# Start MongoDB
docker-compose up -d mongo

# Run the API server
uv run uvicorn api.main:app --reload

# Run tests
uv run pytest -v

# Lint & type-check
uv run ruff check .
uv run mypy .
```

### Health Check

```bash
curl http://localhost:8000/health
# {"status": "healthy", "version": "0.1.0", "db": "connected"}
```

### Quickstart — Freshness Model

```bash
# 1. Download the Fruits Fresh-and-Rotten dataset
uv run python -m data.download

# 2. Extract CLIP ViT-B/32 embeddings (512-dim)
uv run python -m models.freshness.preprocess

# 3. Train the freshness MLP regressor
uv run python -m models.freshness.train

# 4. Run inference on a single image
uv run python -c "
from models.freshness import FreshnessInference
engine = FreshnessInference('data/processed/freshness_best.pt')
result = engine.predict('data/raw/freshapples/some_image.jpg')
print(result)
"
```

### Quickstart -- Portion Estimation

```bash
uv run python -c "
from models.portion import PortionEstimator
est = PortionEstimator()
results = est.estimate('data/raw/freshapples/some_image.jpg')
for r in results:
    print(f'{r.label}: {r.estimated_grams}g +/- {r.uncertainty_grams}g')
"
```

### Quickstart — Agent (Meal Planning)

```python
import asyncio
from agent import run_agent
from db.session import get_database

async def main():
    db = get_database()
    result = await run_agent(user_id="user_001", db=db, meal_type="dinner")
    for recipe in result["selected_plan"]:
        print(f"{recipe['name']} — score: {recipe['_score']:.3f}")
    print(f"Projected: {result['projected_macros']}")

asyncio.run(main())
```

See `notebooks/04_agent_demo.ipynb` for a full walkthrough that runs
entirely in-memory (no MongoDB required).

## Project Structure

```
NutriScan/
├── api/                  # FastAPI application
│   ├── main.py           # App factory, lifespan, /health
│   └── routes/           # Route modules (future phases)
├── agent/                # LangGraph meal-planning agent
├── db/                   # Database layer
│   ├── models.py         # Pydantic v2 document schemas
│   └── session.py        # AsyncMongoClient manager
├── models/               # ML model definitions
│   ├── freshness/        # CLIP -> freshness MLP (Phase 1)
│   │   ├── dataset.py    # FreshnessDataset (fresh/rotten labels)
│   │   ├── extractor.py  # CLIPExtractor (frozen ViT-B/32)
│   │   ├── model.py      # FreshnessRegressor (MLP + MC Dropout)
│   │   ├── train.py      # Training script
│   │   ├── inference.py  # FreshnessInference entry point
│   │   └── preprocess.py # Batch CLIP embedding extraction
│   └── portion/          # Bbox -> gram estimator (Phase 2)
│       ├── categories.py # FoodMeta lookup table (10 categories)
│       ├── reference.py  # Reference object calibration
│       ├── estimator.py  # PortionEstimator (YOLO + geometry)
│       └── pipeline.py   # PortionPipeline (freshness + grams)
├── data/                 # Datasets (git-ignored)
│   ├── raw/
│   └── processed/
├── tests/                # pytest test suite
│   ├── unit/
│   └── integration/
├── notebooks/            # Jupyter exploration
├── Dockerfile            # Multi-stage build
├── docker-compose.yml    # App + MongoDB services
└── pyproject.toml        # uv-managed dependencies
```

## Phase Checklist

| Phase | Description | Status |
|-------|------------|--------|
| **0** | Project scaffold, CI, Docker, schemas, health endpoint | ✅ Complete |
| **1** | Freshness regression model (CLIP features → expiry) | ✅ Complete |
| **2** | Portion estimation pipeline (bbox -> grams) | ✅ Complete |
| **3** | LangGraph agent (macro tracking, recipe scoring) | ✅ Complete |
| **4** | Full API routes + MongoDB CRUD | ✅ Complete |
| **5** | Integration tests, notebook demo, documentation | ⬜ Pending |

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health check |
| POST | `/fridge/analyse` | Analyse a fridge image |
| POST | `/plan/daily` | Generate a daily meal plan |

### curl examples

```bash
# Health check
curl http://localhost:8000/health

# Analyse a fridge image
curl -X POST http://localhost:8000/fridge/analyse \
  -F "image=@path/to/fridge.jpg" \
  -F "user_id=user_001"

# Generate a daily meal plan
curl -X POST http://localhost:8000/plan/daily \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_001", "meal_type": "dinner"}'
```

## License

MIT
