"""
Core environment for AI Disaster Relief Logistics & Supply Allocation.

Implements the OpenEnv interface:
  reset()  → EnvironmentState
  step()   → (reward, done, info)
  state()  → EnvironmentState
"""

from __future__ import annotations

import copy
import logging
import random
from typing import Any, Dict, Optional, Tuple

from env.grader import DisasterReliefGrader
from env.models import (
    AgentAction,
    Delivery,
    EnvironmentState,
    RegionNeeds,
    ResourcePool,
    ResourceType,
)
from env.tasks import BaseTask, get_task

logger = logging.getLogger(__name__)


class DisasterReliefEnv:
    """
    OpenEnv-compliant disaster relief logistics simulation.

    The agent must allocate limited resources (food, water, medicine) across
    multiple disaster-affected regions over a fixed number of time steps.

    Objective:
        - Prioritize high-severity regions
        - Reduce total unmet needs
        - Avoid ignoring critical regions
    """

    def __init__(self, task_id: str = "easy", seed: int = 42) -> None:
        self.task_id = task_id
        self._task: Optional[BaseTask] = None
        self._state: Optional[EnvironmentState] = None
        self._grader = DisasterReliefGrader()
        self._seed = seed
        self._rng = random.Random(seed)

        # Tracking for grader
        self._initial_unmet_totals: Dict[str, float] = {}
        self._total_resources_available: Dict[str, float] = {}
        self._total_resources_used: Dict[str, float] = {
            "food": 0.0,
            "water": 0.0,
            "medicine": 0.0,
        }
        self._prev_unmet_total: float = -1.0  # tracks previous-step total for grader bonus

    # ------------------------------------------------------------------
    # OpenEnv Interface
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None) -> EnvironmentState:
        """
        Reset the environment to the initial state.

        Args:
            task_id: Optional task override. Defaults to the constructor value.

        Returns:
            Initial EnvironmentState.
        """
        if task_id:
            self.task_id = task_id

        self._task = get_task(self.task_id)
        task_cfg = self._task.get_config()

        resources = self._task.get_initial_resources()
        regions = self._task.get_regions()

        self._state = EnvironmentState(
            time_step=0,
            max_steps=task_cfg["max_steps"],
            resources=resources,
            regions=regions,
            task_id=self.task_id,
            done=False,
        )
        self._state.recompute_unmet_totals()

        # Record initial conditions for grader
        self._initial_unmet_totals = {
            "food": self._state.unmet_needs_total.food,
            "water": self._state.unmet_needs_total.water,
            "medicine": self._state.unmet_needs_total.medicine,
        }
        self._total_resources_available = {
            "food": resources.food,
            "water": resources.water,
            "medicine": resources.medicine,
        }
        self._total_resources_used = {"food": 0.0, "water": 0.0, "medicine": 0.0}
        self._prev_unmet_total = -1.0
        self._rng = random.Random(self._seed)  # reproducible RNG per episode

        logger.info(
            "Environment reset | task=%s | regions=%d | max_steps=%d",
            self.task_id,
            len(regions),
            task_cfg["max_steps"],
        )
        return copy.deepcopy(self._state)

    def step(
        self, action: AgentAction
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Execute one time step.

        Args:
            action: AgentAction containing a list of deliveries.

        Returns:
            (reward, done, info) tuple.
              - reward: float in [-0.3, 1.0]
              - done:   bool
              - info:   dict with 'state', 'step_details', 'errors'
        """
        if self._state is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")

        if self._state.done:
            raise RuntimeError("Episode is already done. Call reset() to start a new episode.")

        errors = []
        step_fulfilled: Dict[str, float] = {"food": 0.0, "water": 0.0, "medicine": 0.0}

        # Snapshot unmet total BEFORE deliveries (for progress reward shaping)
        unmet_before = sum(r.unmet_needs.total() for r in self._state.regions)

        # ---- Fuel constraint: cost = num_deliveries * 5 litres --------
        fuel_cost = len(action.deliveries) * 5.0
        if fuel_cost > self._state.resources.fuel:
            errors.append(
                f"Insufficient fuel ({self._state.resources.fuel:.1f}L) for "
                f"{len(action.deliveries)} deliveries (need {fuel_cost:.1f}L). "
                "Deliveries reduced to what fuel permits."
            )
            # Allow as many deliveries as fuel supports
            max_deliveries = int(self._state.resources.fuel // 5)
            action.deliveries = action.deliveries[:max_deliveries]
            fuel_cost = max_deliveries * 5.0

        # Deduct fuel
        self._state.resources.fuel = max(0.0, self._state.resources.fuel - fuel_cost)

        # ---- Process deliveries ----------------------------------------
        region_map = {r.id: r for r in self._state.regions}

        for delivery in action.deliveries:
            errors += self._process_delivery(
                delivery, region_map, step_fulfilled
            )

        # Snapshot unmet total AFTER deliveries
        unmet_after_delivery = sum(r.unmet_needs.total() for r in self._state.regions)

        # ---- Time progression ------------------------------------------
        self._state.time_step += 1

        # Dynamic escalation for hard tasks
        if self._task and self._task.escalate_needs:
            for region in self._state.regions:
                region.escalate_needs(self._task.escalation_factor)

        # ---- Stochastic severity drift every 2 steps -------------------
        # Simulates evolving disaster conditions (deterministic via seeded RNG).
        # Every 2nd step there is a 40% chance each region's severity bumps +1.
        if self._state.time_step % 2 == 0:
            for region in self._state.regions:
                if self._rng.random() < 0.40:
                    region.severity = min(10, region.severity + 1)
                    logger.debug(
                        "Severity drift | region=%s | new_severity=%d",
                        region.id, region.severity,
                    )

        # ---- Compute deaths per region ---------------------------------
        total_deaths_this_step = 0
        for region in self._state.regions:
            d = region.compute_deaths()
            region.deaths_this_step = d
            region.total_deaths += d
            total_deaths_this_step += d

        # ---- Update global unmet totals --------------------------------
        self._state.recompute_unmet_totals()

        # ---- Check done ------------------------------------------------
        done = self._state.time_step >= self._state.max_steps
        self._state.done = done

        # ---- Compute step reward (smooth shaping) ----------------------
        reward = self._step_reward(
            step_fulfilled, total_deaths_this_step,
            unmet_before, unmet_after_delivery, fuel_cost,
        )

        # ---- Compute final score if episode over -----------------------
        if done:
            final_score = self._grader.compute_score(
                state=self._state,
                initial_unmet_totals=self._initial_unmet_totals,
                total_resources_available=self._total_resources_available,
                total_resources_used=self._total_resources_used,
                prev_unmet_total=self._prev_unmet_total,
            )
            self._state.episode_score = max(0.01, min(0.99, float(final_score)))
            breakdown = self._grader.score_breakdown(
                state=self._state,
                initial_unmet_totals=self._initial_unmet_totals,
                total_resources_available=self._total_resources_available,
                total_resources_used=self._total_resources_used,
                prev_unmet_total=self._prev_unmet_total,
            )
            logger.info("Episode complete | score=%.4f | breakdown=%s", final_score, breakdown)
        else:
            breakdown = {}

        # Update prev unmet for next step
        self._prev_unmet_total = self._state.unmet_needs_total.total()

        info = {
            "state": copy.deepcopy(self._state).model_dump(),
            "step_fulfilled": step_fulfilled,
            "errors": errors,
            "score_breakdown": breakdown,
        }

        return reward, done, info

    def state(self) -> EnvironmentState:
        """Return the current environment state (read-only copy)."""
        if self._state is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        return copy.deepcopy(self._state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_delivery(
        self,
        delivery: Delivery,
        region_map: Dict,
        step_fulfilled: Dict[str, float],
    ) -> list:
        """Validate and apply a single delivery. Returns list of error strings."""
        errors = []

        # Validate region
        if delivery.region_id not in region_map:
            errors.append(f"Unknown region_id: {delivery.region_id}")
            return errors

        region = region_map[delivery.region_id]
        resource = delivery.resource
        amount = delivery.amount

        # Validate amount > 0
        if amount <= 0:
            errors.append(
                f"Delivery to {delivery.region_id}: amount must be positive, got {amount}"
            )
            return errors

        # Clamp to available resources (action validation)
        available = self._state.resources.available(resource)
        if amount > available:
            logger.warning(
                "Clamping delivery of %.2f %s to available %.2f",
                amount, resource.value, available,
            )
            amount = available
            errors.append(
                f"Insufficient {resource.value}: requested {delivery.amount:.2f}, "
                f"available {available:.2f}. Clamped to prevent over-allocation."
            )

        if amount <= 0:
            errors.append(f"No {resource.value} available for delivery to {delivery.region_id}.")
            return errors

        # Deduct from global pool
        self._state.resources.deduct(resource, amount)
        self._total_resources_used[resource.value] += amount

        # Reduce unmet need in region
        fulfilled = region.apply_delivery(resource, amount)
        step_fulfilled[resource.value] += fulfilled

        logger.debug(
            "Delivery | region=%s | resource=%s | amount=%.2f | fulfilled=%.2f",
            delivery.region_id, resource.value, amount, fulfilled,
        )
        return errors

    def _step_reward(
        self,
        step_fulfilled: Dict[str, float],
        deaths_this_step: int,
        unmet_before: float,
        unmet_after_delivery: float,
        fuel_cost: float,
    ) -> float:
        """
        Smooth per-step reward with multiple shaping components:

          +0.001 per unit of unmet need fulfilled this step
          +0.002 incremental bonus per unit of unmet-needs reduction vs pre-delivery
          -0.0001 per death this step
          -0.0005 per litre of fuel wasted on zero-fulfillment deliveries (light penalty)
        """
        fulfillment_reward = sum(step_fulfilled.values()) * 0.001

        # Incremental improvement bonus: reward net reduction after delivery
        improvement = max(0.0, unmet_before - unmet_after_delivery)
        incremental_bonus = improvement * 0.002

        death_penalty = deaths_this_step * 0.0001

        # Light waste penalty: if fuel was spent but nothing was fulfilled
        total_fulfilled = sum(step_fulfilled.values())
        waste_penalty = (fuel_cost * 0.0005) if (fuel_cost > 0 and total_fulfilled == 0) else 0.0

        reward = fulfillment_reward + incremental_bonus - death_penalty - waste_penalty
        # STRICT CLAMP: OpenEnv constraints often require reward in [0, 1] range
        reward = max(0.0, min(1.0, float(reward)))
        return round(reward, 6)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_task_config(self) -> Optional[Dict]:
        if self._task:
            return self._task.get_config()
        return None
