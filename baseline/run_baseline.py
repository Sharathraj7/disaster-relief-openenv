"""
Improved heuristic baseline agent for the AI Disaster Relief Logistics environment.

Strategy:
  1. Sort regions by a combined score: severity × unmet_need_for_resource
     (prioritises both urgency AND gap size simultaneously)
  2. Allocate resources proportionally to combined weight — not fixed amounts
  3. Respect fuel constraints: each delivery costs 5L of fuel
  4. Stop when resources are exhausted or episode ends

Produces a deterministic, reproducible baseline score.

Usage:
  python -m baseline.run_baseline --task easy
  python -m baseline.run_baseline --task medium
  python -m baseline.run_baseline --task hard
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.environment import DisasterReliefEnv
from env.models import AgentAction, Delivery, Region, ResourceType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("baseline")


# ---------------------------------------------------------------------------
# Heuristic Agent
# ---------------------------------------------------------------------------

class HeuristicAgent:
    """
    Improved rule-based heuristic agent.

    Allocation weight = severity × unmet_need_for_resource
    This jointly captures both urgency (severity) and gap size (unmet need),
    producing smarter allocations than severity-only weighting.

    Fuel constraint: each delivery line consumes 5L of fuel. The agent trims
    its delivery plan to fit within available fuel.
    """

    RESOURCE_TYPES = [ResourceType.FOOD, ResourceType.WATER, ResourceType.MEDICINE]
    FUEL_PER_DELIVERY = 5.0

    def select_action(self, state) -> AgentAction:
        """
        Generate an action for the current environment state.

        Policy:
          - For each resource type:
              * Compute weight = severity × unmet_need per region
              * Allocate proportionally by weight, capped at actual unmet need
          - Trim deliveries to fuel budget
        """
        deliveries: List[Delivery] = []

        for resource in self.RESOURCE_TYPES:
            available = state.resources.available(resource)
            if available <= 0:
                continue

            # Build (region, weight) pairs — combined severity×need score
            weighted_regions = []
            for r in state.regions:
                unmet = getattr(r.unmet_needs, resource.value)
                if unmet > 0:
                    weight = r.severity * unmet  # combined score
                    weighted_regions.append((r, weight, unmet))

            if not weighted_regions:
                continue

            total_weight = sum(w for _, w, _ in weighted_regions)
            if total_weight == 0:
                continue

            budget_remaining = available
            for region, weight, unmet in sorted(weighted_regions, key=lambda x: x[1], reverse=True):
                # Proportional share of available, but never more than actual unmet
                proportional_alloc = available * (weight / total_weight)
                alloc = round(min(proportional_alloc, unmet, budget_remaining), 2)

                if alloc > 0:
                    deliveries.append(
                        Delivery(
                            region_id=region.id,
                            resource=resource,
                            amount=alloc,
                        )
                    )
                    budget_remaining -= alloc

        # ---- Fuel guard: trim deliveries to what fuel allows ----
        fuel_available = state.resources.fuel
        max_deliveries = int(fuel_available // self.FUEL_PER_DELIVERY)
        if len(deliveries) > max_deliveries:
            logger.debug(
                "Trimming %d deliveries to %d due to fuel limit (%.1fL available)",
                len(deliveries), max_deliveries, fuel_available,
            )
            # Keep highest-priority (first) deliveries
            deliveries = deliveries[:max_deliveries]

        return AgentAction(deliveries=deliveries)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_baseline(task_id: str = "easy") -> Dict:
    """
    Run a full baseline episode.

    Returns:
        Dict with score, cumulative_reward, and step_history.
    """
    print(f"\n{'='*60}")
    print(f"  BASELINE HEURISTIC AGENT — Task: {task_id.upper()}")
    print(f"{'='*60}\n")

    env = DisasterReliefEnv(task_id=task_id)
    agent = HeuristicAgent()

    state = env.reset(task_id=task_id)

    cumulative_reward = 0.0
    step_history = []
    step_num = 0

    print(f"Regions: {[r.id for r in state.regions]}")
    print(f"Initial resources: food={state.resources.food}, "
          f"water={state.resources.water}, medicine={state.resources.medicine}\n")

    while not state.done:
        step_num += 1
        action = agent.select_action(state)

        deliveries_summary = [
            f"{d.region_id}:{d.resource.value}={d.amount:.1f}"
            for d in action.deliveries
        ]
        logger.info("Step %d | deliveries: %s", step_num, deliveries_summary)

        reward, done, info = env.step(action)
        cumulative_reward += reward

        state = env.state()

        step_history.append({
            "step": step_num,
            "deliveries": deliveries_summary,
            "reward": round(reward, 6),
            "unmet_total": {
                "food": round(state.unmet_needs_total.food, 2),
                "water": round(state.unmet_needs_total.water, 2),
                "medicine": round(state.unmet_needs_total.medicine, 2),
            },
        })

    # Final results
    final_state = env.state()
    final_score = final_state.episode_score
    breakdown = env._grader.score_breakdown(
        state=final_state,
        initial_unmet_totals=env._initial_unmet_totals,
        total_resources_available=env._total_resources_available,
        total_resources_used=env._total_resources_used,
    )

    _print_results(final_state, cumulative_reward, step_history, breakdown)

    return {
        "task_id": task_id,
        "final_score": final_score,
        "cumulative_reward": round(cumulative_reward, 6),
        "steps": step_num,
        "score_breakdown": breakdown,
        "step_history": step_history,
    }


def _print_results(final_state, cumulative_reward, history, breakdown) -> None:
    """Pretty-print final episode results."""
    score = final_state.episode_score
    unmet = final_state.unmet_needs_total

    print("\n[BASELINE RESULTS]")
    print(f"  Final Score        : {score:.4f} / 1.0000")
    print(f"  Cumulative Reward  : {cumulative_reward:.4f}")
    print(f"  Total Unmet Needs  : Food={unmet.food:.1f} | "
          f"Water={unmet.water:.1f} | Medicine={unmet.medicine:.1f}")

    print("\n[SCORE BREAKDOWN]")
    for k, v in breakdown.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")

    print("\n[REGION SUMMARY]")
    for region in final_state.regions:
        un = region.unmet_needs
        ini = region.initial_needs
        ini_total = ini.food + ini.water + ini.medicine
        remaining = un.food + un.water + un.medicine
        pct_met = (1 - remaining / max(1, ini_total)) * 100
        print(
            f"  {region.id} ({region.name}) | Sev={region.severity} | "
            f"Met={pct_met:.1f}% | Deaths={region.total_deaths}"
        )

    print("\n[STEP HISTORY]")
    for h in history:
        print(f"  Step {h['step']}: reward={h['reward']:.4f} | "
              f"deliveries={len(h['deliveries'])} | unmet={h['unmet_total']}")

    grade = "A" if score >= 0.85 else "B" if score >= 0.70 else "C" if score >= 0.55 else "F"
    print(f"\n[BASELINE GRADE]: {grade} (Score: {score:.4f})")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the baseline heuristic agent for the disaster relief environment."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="easy",
        choices=["easy", "medium", "hard", "extreme"],
        help="Task difficulty level (default: easy)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save results as JSON",
    )
    args = parser.parse_args()

    results = run_baseline(task_id=args.task)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_json}")

    return results


if __name__ == "__main__":
    main()
