"""
Inference script for the AI Disaster Relief Logistics Agent.

Uses an OpenAI-compatible LLM (via HuggingFace Router or OpenAI API) to generate
actions against the FastAPI environment server. Runs a full episode and prints the
final score with a detailed breakdown.

Environment Variables (MANDATORY — set before running):
  API_BASE_URL  — LLM API endpoint, e.g. https://router.huggingface.co/v1
                  This is passed as base_url to the OpenAI client.
  MODEL_NAME    — LLM model identifier, e.g. meta-llama/Llama-3.1-70B-Instruct
  HF_TOKEN      — HuggingFace API key (used as the LLM API key)

Optional:
  OPENAI_API_KEY — OpenAI key (fallback if HF_TOKEN is not set)
  ENV_SERVER_URL — URL of the FastAPI environment server (default: http://localhost:7860)
  TASK_ID        — Task difficulty: easy | medium | hard (default: easy)

Usage:
  # With HuggingFace Router:
  API_BASE_URL=https://router.huggingface.co/v1 MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct HF_TOKEN=hf_... python inference.py

  # With OpenAI:
  API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-4o OPENAI_API_KEY=sk-... python inference.py

  # Run all 3 tasks:
  python inference.py --all-tasks
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("inference")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Configuration — matches organizer-prescribed variable semantics:
#   API_BASE_URL = LLM API endpoint  (→ OpenAI client base_url)
#   MODEL_NAME   = LLM model id
#   HF_TOKEN     = HuggingFace / LLM API key  (primary)
# ---------------------------------------------------------------------------

# LLM endpoint — passed as base_url to OpenAI client (required by organizers)
API_BASE_URL: str = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"

# LLM model identifier
MODEL_NAME: str = os.getenv("MODEL_NAME")

# API key: HF_TOKEN takes priority, fall back to API_KEY / OPENAI_API_KEY
API_KEY: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")

# Environment server URL — SEPARATE from API_BASE_URL so the two don't conflict
ENV_SERVER_URL: str = os.getenv("ENV_SERVER_URL") or "http://localhost:7860"

# Episode configuration
TASK_ID: str = os.environ.get("TASK_ID", "easy")
MAX_RETRIES: int = 3

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an AI disaster relief logistics coordinator operating under strict evaluation constraints.

CRITICAL REQUIREMENTS:
- You must output ONLY valid JSON in the exact format shown below.
- You must prioritize minimizing deaths across all regions.
- You must consider resource constraints (food, water, medicine, fuel).
- You must anticipate future escalation (needs increase over time).
- You must NOT output explanations, reasoning, or extra text.
- You must NOT hallucinate resources — never exceed available amounts.
- You must ensure allocations are efficient and non-wasteful.

DECISION RULES:
1. Always prioritize highest severity × unmet need. Severity 8-10 = critical.
2. Spread resources across ALL critical regions — do not over-concentrate on one.
3. Avoid sending resources to low-priority regions if high-priority ones still have unmet needs.
4. Do not exceed available resources. Sum of each resource type across all deliveries must NOT exceed depot amount.
5. Each delivery consumes 5L of fuel. Plan deliveries within your fuel budget.
6. Anticipate worsening conditions — unmet needs may escalate each step.
7. Ensure no critical region (severity >= 8) is completely ignored.

OUTPUT FORMAT (STRICT — any deviation causes failure):
Return ONLY valid JSON, no markdown, no explanation, no comments:
{"deliveries": [
  {"region_id": "R1", "resource": "water", "amount": 80.0},
  {"region_id": "R2", "resource": "food", "amount": 40.0}
]}

FIELD CONSTRAINTS:
- region_id must match exactly (e.g. R1, R2, R3 ...)
- resource must be one of: food, water, medicine
- amount must be a positive number
- deliveries must be a non-empty list

FAILURE CONDITIONS (YOU MUST AVOID):
- Invalid JSON or missing fields
- Over-allocation of any resource beyond depot availability
- Ignoring high-severity regions when resources exist
- Producing empty deliveries list when resources are available
- Any text outside the JSON object
"""


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def reset_environment(task_id: str) -> Dict[str, Any]:
    """Call /reset to initialise the environment."""
    url = f"{ENV_SERVER_URL}/reset?task_id={task_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    logger.info("Environment reset | task=%s", task_id)
    return data


def step_environment(action: Dict) -> Dict[str, Any]:
    """Call POST /step with the given action dict."""
    url = f"{ENV_SERVER_URL}/step"
    payload = {"action": action}
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_current_state() -> Dict[str, Any]:
    """Call GET /state."""
    url = f"{ENV_SERVER_URL}/state"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def build_client() -> OpenAI:
    """
    Build and return an OpenAI-compatible client.

    Follows the organizer's prescribed pattern:
      OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    This allows the same inference.py to work with:
      - HuggingFace Router (API_BASE_URL=https://router.huggingface.co/v1)
      - OpenAI API       (API_BASE_URL=https://api.openai.com/v1)
      - Any OpenAI-compatible endpoint
    """
    if not API_KEY:
        logger.error(
            "No API key found. Set HF_TOKEN or OPENAI_API_KEY environment variable."
        )
        sys.exit(1)
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def state_to_prompt(state: Dict[str, Any]) -> str:
    """Format the environment state into a human-readable prompt."""
    resources = state["resources"]
    regions = state["regions"]
    time_step = state["time_step"]
    max_steps = state["max_steps"]
    unmet = state["unmet_needs_total"]

    lines = [
        f"=== DISASTER RELIEF SITUATION — Step {time_step + 1}/{max_steps} ===",
        "",
        "AVAILABLE RESOURCES (global depot):",
        f"  Food     : {resources['food']:.1f} tons",
        f"  Water    : {resources['water']:.1f} kL",
        f"  Medicine : {resources['medicine']:.1f} units",
        f"  Trucks   : {resources['trucks']}",
        f"  Fuel     : {resources['fuel']:.1f} L",
        "",
        "TOTAL UNMET NEEDS ACROSS ALL REGIONS:",
        f"  Food: {unmet['food']:.1f} | Water: {unmet['water']:.1f} | Medicine: {unmet['medicine']:.1f}",
        "",
        "AFFECTED REGIONS (sorted by severity, highest first):",
    ]

    sorted_regions = sorted(regions, key=lambda r: r["severity"], reverse=True)
    for r in sorted_regions:
        un = r["unmet_needs"]
        lines.append(
            f"\n  [{r['id']}] {r['name']} — Severity: {r['severity']}/10 | "
            f"Population: {r['population']:,} | Type: {r['disaster_type']}"
        )
        lines.append(f"    Location : {r['location']}")
        lines.append(
            f"    Unmet Needs — Food: {un['food']:.1f} | Water: {un['water']:.1f} | "
            f"Medicine: {un['medicine']:.1f}"
        )

    lines += [
        "",
        f"Steps remaining: {max_steps - time_step - 1}",
        "",
        "Decide your resource allocation. Respond with valid JSON ONLY.",
    ]

    return "\n".join(lines)


def call_llm(client: OpenAI, prompt: str, step_num: int) -> Dict:
    """Call the LLM and parse the JSON action. Retries on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("Calling LLM | step=%d | attempt=%d", step_num, attempt)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )

            raw_content = response.choices[0].message.content
            logger.debug("LLM response: %s", raw_content)

            parsed = json.loads(raw_content)

            # Accept both response shapes:
            #   {"deliveries": [...]}            ← new compact format
            #   {"action": {"deliveries": [...]}} ← legacy wrapper format
            if "deliveries" in parsed:
                action = {"deliveries": parsed["deliveries"]}
            elif "action" in parsed and "deliveries" in parsed["action"]:
                action = parsed["action"]
                reasoning = parsed.get("reasoning", "")
                if reasoning:
                    logger.info("LLM reasoning: %s", reasoning)
            else:
                raise ValueError(f"No 'deliveries' key found in response: {parsed}")

            # Basic sanity check
            if not isinstance(action["deliveries"], list):
                raise ValueError("deliveries must be a list")

            return action

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("Parse error on attempt %d/%d: %s", attempt, MAX_RETRIES, e)
            if attempt == MAX_RETRIES:
                logger.error("All retry attempts failed. Using empty action.")
                return {"deliveries": []}
            time.sleep(1)
        except Exception as e:
            logger.error("LLM API error: %s", e)
            if attempt == MAX_RETRIES:
                return {"deliveries": []}
            time.sleep(2)

    return {"deliveries": []}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(task_id: str = TASK_ID) -> None:
    """Run a complete episode using the LLM as the policy."""
    logger.info("=" * 60)
    logger.info("  AI DISASTER RELIEF LOGISTICS AGENT")
    logger.info("  Task: %s | Model: %s", task_id.upper(), MODEL_NAME)
    logger.info("=" * 60)

    # 1. Build LLM client
    client = build_client()

    # 2. Reset environment (single reset — no reset between steps)
    reset_data = reset_environment(task_id)
    state = reset_data["observation"]
    max_steps = state["max_steps"]

    print(f"[START] task={task_id} env=ai-disaster-relief-logistics model={MODEL_NAME}")

    logger.info("Environment initialised with %d regions.", len(state['regions']))
    logger.info("Running for up to %d steps.", max_steps)

    cumulative_reward = 0.0
    step_history: List[Dict] = []
    run_log: List[Dict] = []  # structured log saved to run_log.json

    # 3. Episode loop
    for step_num in range(1, max_steps + 1):
        # Build prompt from current (returned) state
        prompt = state_to_prompt(state)

        # Query LLM for action
        action = call_llm(client, prompt, step_num)

        # Execute action in environment — use state returned by /step, no re-reset
        try:
            result = step_environment(action)
        except requests.HTTPError as e:
            logger.error("Step API error: %s | Response: %s", e, e.response.text if e.response else "N/A")
            break

        reward = result["reward"]
        done = result["done"]
        info = result["info"]
        state = info["state"]  # use state from step response directly
        errors = info.get("errors", [])
        error_str = errors[0] if errors else "null"

        cumulative_reward += reward

        print(f"[STEP] step={step_num} action={json.dumps(action)} reward={reward:.2f} done={str(done).lower()} error={error_str}")

        step_record = {
            "step": step_num,
            "action": action,
            "reward": reward,
            "cumulative_reward": round(cumulative_reward, 6),
            "errors": errors,
            "unmet_needs_total": state.get("unmet_needs_total", {}),
            "resources_remaining": {
                k: state["resources"][k]
                for k in ["food", "water", "medicine", "fuel"]
            },
        }
        step_history.append(step_record)
        run_log.append(step_record)

        if done:
            print(f"[END] success=true steps={step_num} rewards={cumulative_reward:.2f}")
            break
    else:
        # Loop ended without done=True (edge case)
        print(f"[END] success=true steps={max_steps} rewards={cumulative_reward:.2f}")

    # 4. Save run log to JSON
    log_path = "run_log.json"
    try:
        with open(log_path, "w") as f:
            json.dump(
                {
                    "task_id": task_id,
                    "model": MODEL_NAME,
                    "final_score": state.get("episode_score", 0.0),
                    "cumulative_reward": round(cumulative_reward, 6),
                    "steps": run_log,
                },
                f,
                indent=2,
            )
        logger.info("Run log saved to %s", log_path)
    except OSError as e:
        logger.warning("Could not save run log: %s", e)

    # 5. Print final results
    _print_final_results(state, cumulative_reward, step_history)


def _print_final_results(
    final_state: Dict, cumulative_reward: float, history: List[Dict]
) -> None:
    """Log the final episode summary to stderr (logger) to keep stdout clean."""
    score = final_state.get("episode_score", 0.0)
    unmet = final_state.get("unmet_needs_total", {})

    logger.info("FINAL RESULTS")
    logger.info("  Episode Score      : %.4f / 1.0000", score)
    logger.info("  Cumulative Reward  : %.4f", cumulative_reward)
    logger.info("  Total Unmet Needs  : Food=%.1f | Water=%.1f | Medicine=%.1f",
                unmet.get('food', 0), unmet.get('water', 0), unmet.get('medicine', 0))

    logger.info("REGION SUMMARY")
    for region in final_state.get("regions", []):
        un = region["unmet_needs"]
        ini = region["initial_needs"]
        ini_total = ini.get("food", 0) + ini.get("water", 0) + ini.get("medicine", 0)
        remaining = un.get("food", 0) + un.get("water", 0) + un.get("medicine", 0)
        pct_met = (1 - remaining / max(1, ini_total)) * 100
        logger.info("  %s (%s) | Severity: %s | Needs Met: %.1f%% | Deaths: %s",
                    region['id'], region['name'], region['severity'], pct_met, region.get('total_deaths', 0))

    logger.info("STEP HISTORY")
    for h in history:
        deliveries = h["action"].get("deliveries", [])
        logger.info("  Step %d: reward=%.4f | deliveries=%d", h['step'], h['reward'], len(deliveries))

    grade = "A" if score >= 0.85 else "B" if score >= 0.70 else "C" if score >= 0.45 else "F"
    logger.info("FINAL GRADE: %s (Score: %.4f)", grade, score)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--all-tasks":
        # Run all 3 tasks sequentially and print a combined summary
        logger.info("=" * 60)
        logger.info("  RUNNING ALL 3 TASKS")
        logger.info("=" * 60)
        all_scores = {}
        for t in ["easy", "medium", "hard"]:
            run_episode(task_id=t)
            all_scores[t] = None  # scores are printed per-episode
        logger.info("All tasks complete. Check individual episode summaries above.")
    else:
        task = sys.argv[1] if len(sys.argv) > 1 else TASK_ID
        run_episode(task_id=task)
