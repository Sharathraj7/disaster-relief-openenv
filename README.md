---
title: AI Disaster Relief Logistics
emoji: 🌊
colorFrom: blue
colorTo: red
sdk: docker
app_file: app.py
pinned: false
---

# AI Disaster Relief Logistics & Supply Allocation Agent

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v1.0-blue)](https://openenv.ai)
[![Python](https://img.shields.io/badge/python-3.10+-green)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌊 Problem Description

Natural disasters — floods, earthquakes, and cyclones — create sudden, overwhelming demand for emergency resources. Relief agencies must make rapid decisions about **where to send food, water, and medicine** when supplies are limited and every hour counts.

This environment simulates exactly that challenge. An AI agent must act as a **logistics coordinator**, deciding how to route scarce supplies to multiple affected regions under time pressure, with incomplete coverage, and escalating human need.

---

## 🌍 Real-World Use Case

| Domain | Application |
|--------|------------|
| Humanitarian Aid | UN OCHA, Red Cross resource routing |
| Government | National Disaster Management Authority (NDMA) decision support |
| NGOs | Médecins Sans Frontières supply-chain optimization |
| Research | Benchmarking AI planning agents under resource scarcity |

---

## 🏗️ Project Structure

```
├── openenv.yaml            # OpenEnv specification
├── env/
│   ├── environment.py      # Core environment (reset/step/state)
│   ├── models.py           # Pydantic typed models
│   ├── tasks.py            # Easy / Medium / Hard task configs
│   └── grader.py           # Deterministic scoring system
├── baseline/
│   └── run_baseline.py     # Heuristic baseline agent
├── inference.py            # LLM-powered agent (OpenAI client)
├── app.py                  # FastAPI server
├── Dockerfile              # Container definition
├── requirements.txt        # Dependencies
└── README.md
```

---

## 📐 Observation Space

The agent observes the full environment state at each step:

```json
{
  "time_step": 2,
  "max_steps": 8,
  "resources": {
    "food": 180.5,
    "water": 240.0,
    "medicine": 90.0,
    "trucks": 8,
    "fuel": 1200.0
  },
  "regions": [
    {
      "id": "R1",
      "name": "City Center",
      "severity": 9,
      "population": 80000,
      "disaster_type": "earthquake",
      "location": "Urban core, infrastructure destroyed",
      "unmet_needs": {
        "food": 75.0,
        "water": 130.0,
        "medicine": 55.0
      }
    }
  ],
  "unmet_needs_total": {
    "food": 145.0,
    "water": 280.0,
    "medicine": 100.0
  }
}
```

---

## 🎯 Action Space

The agent submits a list of deliveries per step:

```json
{
  "action": {
    "deliveries": [
      {"region_id": "R1", "resource": "water",    "amount": 80.0},
      {"region_id": "R1", "resource": "food",     "amount": 50.0},
      {"region_id": "R2", "resource": "medicine", "amount": 30.0}
    ]
  }
}
```

**Constraints:**
- `resource` must be one of: `food`, `water`, `medicine`
- `amount` must be ≥ 0
- Total allocated per resource cannot exceed the depot's available amount
- Excess allocations are automatically clamped

---

## 🏆 Reward Design

### Episode Score (0.0 – 1.0)

| Component | Weight | Description |
|-----------|--------|-------------|
| Priority Allocation | 0.50 | Severity-weighted coverage rate across all regions |
| Resource Efficiency | 0.30 | Impact per unit resource used (lenient under scarcity) |
| Unmet Needs Reduction | 0.20 | Overall % of total needs fulfilled |
| **Progress Bonus** | **+0.10** | Awarded when unmet needs decreased vs previous step |
| **Critical Region Penalty** | **−0.15** | If any severity ≥ 8 region ends with >70% unmet needs |

### Step Reward

Each step returns a shaped intermediate reward:
```
reward = (fulfilled_needs × 0.001)
        + (improvement_vs_pre_delivery × 0.002)
        - (deaths_this_step × 0.0001)
        - (fuel_wasted × 0.0005)   # only when 0 needs were fulfilled
```

### Performance Grades

| Grade | Score Range |
|-------|-------------|
| 🏆 A  | ≥ 0.85      |
| 🥈 B  | 0.70 – 0.84 |
| 🥉 C  | 0.45 – 0.69 |
| ❌ F  | < 0.45      |

---

## 📋 Tasks

| Task | Regions | Resources | Max Steps | Difficulty |
|------|---------|-----------|-----------|------------|
| `easy` | 2 (flood) | Sufficient | 5 | Learning |
| `medium` | 4 (earthquake) | Limited | 8 | Moderate |
| `hard` | 6 (cyclone) | Severe scarcity | 12 | Expert |

**Hard task** includes dynamic need escalation: unmet needs grow by 8% per step.

---

## 🌀 Stochastic Dynamics

To simulate evolving disaster conditions realistically:

- **Every 2 steps**, each region has a **40% chance** of its severity increasing by +1 (capped at 10)
- This models aftershocks, secondary flooding, disease spread, etc.
- The RNG is **seeded deterministically** (default `seed=42`) so episodes are reproducible
- Agents must adapt over time — a region that was severity 6 may become severity 7 mid-episode

To run with a different seed:
```python
env = DisasterReliefEnv(task_id="hard", seed=123)
```

---

## ⛽ Fuel Constraint

Each delivery truck run consumes **5 litres of fuel**:

```
fuel_consumed = number_of_deliveries × 5
```

- If the agent requests more deliveries than fuel allows, deliveries are **trimmed** (not rejected silently)
- A warning is added to the step `errors` list
- An agent that over-plans wastes potential: **plan within your fuel budget!**
- Fuel is **not replenished** between steps

| Task | Starting Fuel | Max Deliveries/Step (full fuel) |
|------|-------------|----------------------------------|
| easy | 2000 L | 400 |
| medium | 1500 L | 300 |
| hard | 1000 L | 200 |

---

## 📖 Example Episode Walkthrough

**Task: Medium (Earthquake, 4 regions, 8 steps)**

**Initial state:**
```
Resources: food=300, water=400, medicine=150
R1 City Center  (severity=9): needs food=120, water=200, medicine=80
R2 Industrial   (severity=6): needs food=60,  water=90,  medicine=35
R3 Mountain     (severity=7): needs food=80,  water=100, medicine=50
R4 Suburban     (severity=3): needs food=40,  water=60,  medicine=20
```

**Step 1 — Agent prioritises R1 and R3 (highest severity):**
```json
{"deliveries": [
  {"region_id": "R1", "resource": "water",    "amount": 160},
  {"region_id": "R1", "resource": "food",     "amount": 100},
  {"region_id": "R3", "resource": "medicine", "amount": 40}
]}
```
Reward: +0.42 (large fulfillment on severity-9 region)

**Step 2 — Remaining needs spread across R2, R3, R4:**
```json
{"deliveries": [
  {"region_id": "R1", "resource": "medicine", "amount": 80},
  {"region_id": "R2", "resource": "water",    "amount": 90},
  {"region_id": "R3", "resource": "food",     "amount": 80}
]}
```
Reward: +0.35

**Final score breakdown:**
```
Priority Allocation  : 0.92  × 0.50 = 0.46
Resource Efficiency  : 0.88  × 0.30 = 0.26
Unmet Needs Reduction: 0.95  × 0.20 = 0.19
Progress Bonus       :              + 0.10
Critical Penalty     :              − 0.00
─────────────────────────────────────────
Final Score          :              = 0.91  (Grade A)
```

---

## 📊 Expected Score Ranges

| Task | Random Agent | Baseline Heuristic | Strong LLM Agent |
|------|-------------|-------------------|------------------|
| easy | 0.10 – 0.30 | 0.90 – 1.00 | 0.95 – 1.00 |
| medium | 0.05 – 0.20 | 0.75 – 0.95 | 0.85 – 0.98 |
| hard | 0.00 – 0.10 | 0.30 – 0.55 | 0.50 – 0.75 |

---

## 🚀 Setup Instructions

### Local Development

```bash
# 1. Clone the repository
git clone https://huggingface.co/spaces/<your-username>/ai-disaster-relief

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the FastAPI server
python app.py

# 4. Run baseline agent (no API key needed)
python -m baseline.run_baseline --task medium

# 5. Run LLM inference agent
export OPENAI_API_KEY=sk-...
export TASK_ID=medium
python inference.py
```

### Docker

```bash
# Build
docker build -t disaster-relief-env .

# Run
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=sk-... \
  -e TASK_ID=easy \
  disaster-relief-env
```

### Hugging Face Spaces

Deploy directly to Hugging Face Spaces with Docker SDK:
- Set `OPENAI_API_KEY` in Space Secrets
- The server will auto-start on port 7860

---

## 🌐 API Reference

### `GET /`
Health check.

```json
{"status": "ok", "version": "1.0.0", "environment": "ai-disaster-relief-logistics"}
```

---

### `GET /reset?task_id=easy`
Reset the environment. Returns full initial state.

```bash
curl http://localhost:7860/reset?task_id=medium
```

---

### `POST /reset`
Reset with a JSON body.

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "hard"}'
```

---

### `POST /step`
Execute one action step.

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "deliveries": [
        {"region_id": "R1", "resource": "water", "amount": 80},
        {"region_id": "R1", "resource": "food", "amount": 50},
        {"region_id": "R2", "resource": "medicine", "amount": 25}
      ]
    }
  }'
```

**Response:**
```json
{
  "reward": 0.155,
  "done": false,
  "info": {
    "state": { "time_step": 1, "max_steps": 8, "resources": {...}, "regions": [...] },
    "step_fulfilled": {"food": 50.0, "water": 80.0, "medicine": 25.0},
    "errors": [],
    "score_breakdown": {}
  }
}
```

---

### `GET /state`
Get current state without advancing simulation.

---

### `GET /tasks`
List all available tasks.

---

## 📊 Baseline Results

Measured scores from `HeuristicAgent` (severity×unmet-need weighted, seed=42):

| Task | Baseline Score | Grade |
|------|---------------|-------|
| easy | **0.9400** | 🏆 A |
| medium | **0.8662** | 🏆 A |
| hard | **0.4466** | 🥉 C |

Run to reproduce:
```bash
python -m baseline.run_baseline --task easy
python -m baseline.run_baseline --task medium
python -m baseline.run_baseline --task hard
```

A strong LLM agent (e.g. GPT-4o) is expected to reach **0.85–0.98** on easy/medium and **0.50–0.75** on hard.


---

## 🔬 OpenEnv Compliance

This environment implements the full OpenEnv interface:

| Interface | Status |
|-----------|--------|
| `reset()` | ✅ |
| `step(action)` | ✅ |
| `state()` | ✅ |
| Pydantic typed models | ✅ |
| `openenv.yaml` with observation/action spaces | ✅ |
| Three task difficulty levels | ✅ |
| Deterministic grader | ✅ |
| Baseline agent | ✅ |
| Inference script | ✅ |
| FastAPI server | ✅ |
| Docker deployment | ✅ |

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
