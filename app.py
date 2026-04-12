"""
FastAPI server exposing the OpenEnv interface for the
AI Disaster Relief Logistics environment.

Endpoints:
  GET  /            → Health check
  GET  /reset       → Reset environment (defaults to 'easy') 
  POST /reset       → Reset environment with task_id body
  POST /step        → Execute one action step
  GET  /state       → Get current state
  GET  /tasks       → List available tasks
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env.environment import DisasterReliefEnv
from env.models import AgentAction, Delivery, HealthResponse, ResourceType

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("app")


# ---------------------------------------------------------------------------
# Global environment instance
# ---------------------------------------------------------------------------

env: Optional[DisasterReliefEnv] = None
env_lock: asyncio.Lock = asyncio.Lock()  # Protects env from concurrent mutation


@asynccontextmanager
async def lifespan(app: FastAPI):
    global env
    logger.info("Initialising DisasterReliefEnv...")
    env = DisasterReliefEnv(task_id="easy")
    env.reset()
    logger.info("Environment ready.")
    yield
    logger.info("Shutting down environment.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Disaster Relief Logistics & Supply Allocation Agent",
    description=(
        "OpenEnv-compliant simulation where an AI agent allocates limited resources "
        "(food, water, medicine) across disaster-affected regions.\n\n"
        "The agent must prioritize high-severity regions, optimize resource usage, "
        "and minimize unmet needs and casualties."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy"


class StepRequest(BaseModel):
    action: AgentAction


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Environment"])
def health():
    return {"status": "healthy"}


@app.get("/metadata", tags=["Environment"])
def get_metadata():
    return {
        "name": "ai-disaster-relief-logistics",
        "description": "OpenEnv-compliant simulation where an AI agent allocates limited resources across disaster-affected regions.",
    }


@app.get("/schema", tags=["Environment"])
def get_schema():
    from env.models import AgentAction, EnvironmentState

    return {
        "action": AgentAction.model_json_schema(),
        "observation": EnvironmentState.model_json_schema(),
        "state": EnvironmentState.model_json_schema(),
    }


@app.post("/mcp", tags=["Environment"])
async def mcp_endpoint():
    # Minimal mock for the platform validator
    return {"jsonrpc": "2.0", "result": {}, "id": 1}


@app.get("/reset", tags=["Environment"])
async def reset_get(task_id: str = Query(default="easy", description="Task difficulty: easy | medium | hard")):
    global env
    async with env_lock:
        try:
            state = env.reset(task_id=task_id)
            logger.info("Environment reset via GET | task=%s", task_id)
            return {
                "observation": state.model_dump(),
                "reward": 0.0,
                "done": False,
                "info": {
                    "message": "Environment reset successfully",
                    "task_id": task_id,
                },
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("Unexpected error during reset")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset", tags=["Environment"])
async def reset_post(request: Optional[ResetRequest] = None):
    global env
    task_id = request.task_id if request else "easy"
    async with env_lock:
        try:
            state = env.reset(task_id=task_id)
            logger.info("Environment reset via POST | task=%s", task_id)
            return {
                "observation": state.model_dump(),
                "reward": 0.0,
                "done": False,
                "info": {
                    "message": "Environment reset successfully",
                    "task_id": task_id,
                },
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("Unexpected error during reset")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", tags=["Environment"])
async def step(request: StepRequest):
    global env
    async with env_lock:
        try:
            reward, done, info = env.step(request.action)
            observation = info.pop("state", {})
            return {
                "observation": observation,
                "reward": reward,
                "done": done,
                "info": info,
            }
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("Unexpected error during step")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", tags=["Environment"])
async def get_state():
    """Get the current environment state without advancing the simulation."""
    global env
    try:
        state = env.state()
        return {"state": state.model_dump()}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error fetching state")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks", tags=["Environment"])
async def list_tasks():
    """List all available task configurations."""
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "Two-Region Flood Relief",
                "difficulty": "easy",
                "max_steps": 5,
                "description": (
                    "A flood affects 2 regions with sufficient resources. "
                    "Learn basic severity-based prioritization."
                ),
            },
            {
                "id": "medium",
                "name": "Four-Region Earthquake Response",
                "difficulty": "medium",
                "max_steps": 8,
                "description": (
                    "An earthquake strikes 4 regions with limited supplies. "
                    "Triage and allocate scarce resources wisely."
                ),
            },
            {
                "id": "hard",
                "name": "Six-Region Cyclone with Dynamic Deterioration",
                "difficulty": "hard",
                "max_steps": 12,
                "description": (
                    "A cyclone ravages 6+ regions with severe scarcity. "
                    "Unmet needs escalate each step — adapt dynamically."
                ),
            },
            {
                "id": "extreme",
                "name": "Eight-Region Multi-Disaster Challenge",
                "difficulty": "extreme",
                "max_steps": 15,
                "description": (
                    "A catastrophic multi-disaster scenario affecting 8 regions. "
                    "Extreme scarcity and growth of needs. Max logistic complexity."
                ),
            },
        ]
    }


@app.get("/docs-info", tags=["Info"])
async def docs_info():
    """Human-readable API usage instructions."""
    return {
        "usage": {
            "1_reset": "GET /reset?task_id=easy  OR  POST /reset {\"task_id\": \"medium\"}",
            "2_step": "POST /step {\"action\": {\"deliveries\": [{\"region_id\": \"R1\", \"resource\": \"water\", \"amount\": 50}]}}",
            "3_state": "GET /state",
            "4_tasks": "GET /tasks",
            "5_validate": "GET /validate",
        },
        "resources": ["food", "water", "medicine"],
        "task_ids": ["easy", "medium", "hard"],
    }


@app.get("/validate", tags=["Info"])
async def validate_spec():
    """
    Return the parsed openenv.yaml spec.
    Used by OpenEnv validators and automated spec compliance checks.
    """
    spec_path = os.path.join(os.path.dirname(__file__), "openenv.yaml")
    try:
        with open(spec_path, "r") as f:
            spec = yaml.safe_load(f)
        return {"valid": True, "spec": spec}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="openenv.yaml not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
