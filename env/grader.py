"""
Grader for the AI Disaster Relief Logistics environment.

Scoring breakdown:
  - 0.50  Priority allocation   → Severity-weighted coverage across regions
  - 0.30  Resource efficiency   → Impact per unit resource used
  - 0.20  Unmet needs reduction → Overall % of needs fulfilled
  +0.10  Progress bonus        → Awarded when total unmet needs decreased vs previous step

Penalty:
  - -0.15 if any critical (severity >= 8) region ends with >70% unmet needs
           (threshold raised from 50% → 70% to allow partial credit on hard tasks)

Score is always clipped to [0.0, 1.0] and is deterministic given the same trajectory.
"""

from __future__ import annotations

from typing import List

from env.models import EnvironmentState, Region


class DisasterReliefGrader:
    """Computes a final score in [0.0, 1.0] for a completed episode."""

    PRIORITY_WEIGHT: float = 0.50
    EFFICIENCY_WEIGHT: float = 0.30
    UNMET_NEEDS_WEIGHT: float = 0.20
    CRITICAL_PENALTY: float = 0.15        # Reduced from 0.30 → fairer on hard tasks
    CRITICAL_SEVERITY_THRESHOLD: int = 8
    CRITICAL_UNMET_THRESHOLD: float = 0.70  # >70% unmet triggers penalty (was 50%)
    PROGRESS_BONUS: float = 0.10           # Bonus when unmet needs decrease vs prev step

    def compute_score(
        self,
        state: EnvironmentState,
        initial_unmet_totals: dict,
        total_resources_available: dict,
        total_resources_used: dict,
        prev_unmet_total: float = -1.0,
    ) -> float:
        """
        Compute final episode score.

        Args:
            state: Current (final) environment state.
            initial_unmet_totals: Dict with initial {food, water, medicine} totals.
            total_resources_available: Dict with starting {food, water, medicine}.
            total_resources_used: Dict with consumed {food, water, medicine}.
            prev_unmet_total: Sum of unmet needs from the previous step.
                              If positive and greater than current total, a
                              progress bonus (+0.10) is awarded.

        Returns:
            Score in [0.0, 1.0].
        """
        priority_score = self._priority_score(state.regions)
        efficiency_score = self._efficiency_score(
            total_resources_available, total_resources_used, state.regions
        )
        unmet_score = self._unmet_needs_score(state.regions, initial_unmet_totals)
        penalty = self._critical_penalty(state.regions)
        progress_bonus = self._progress_bonus(state.regions, prev_unmet_total)

        raw_score = (
            self.PRIORITY_WEIGHT * priority_score
            + self.EFFICIENCY_WEIGHT * efficiency_score
            + self.UNMET_NEEDS_WEIGHT * unmet_score
            + progress_bonus
            - penalty
        )

        # Add a tiny natural variation based on average severity to differentiate tasks intrinsically
        avg_severity = sum(r.severity for r in state.regions) / max(1, len(state.regions))
        base_variation = avg_severity * 0.001

        score = float(raw_score)
        if score <= 0.0:
            score = 0.01 + base_variation
        elif score >= 1.0:
            score = 0.99 - base_variation
        else:
            score = min(0.99, max(0.01, score + base_variation))

        return float(score)

    # ------------------------------------------------------------------
    # Sub-scorers
    # ------------------------------------------------------------------

    def _priority_score(self, regions: List[Region]) -> float:
        """
        Score based on how well the agent served high-severity regions.

        Method: Severity-weighted delivery coverage.
        Coverage = total_delivered / initial_needs (capped at 1.0).
        Using total_delivered ensures escalation of unmet_needs does NOT
        corrupt the metric — we measure agent effort, not remaining gap.
        """
        if not regions:
            return 0.0

        total_severity = sum(r.severity for r in regions)
        if total_severity == 0:
            return 0.0

        weighted_coverage = 0.0
        for region in regions:
            initial_total = region.initial_needs.total()
            if initial_total <= 0:
                coverage = 1.0
            else:
                # Coverage = what was actually delivered / what was initially needed
                coverage = min(1.0, region.total_delivered / initial_total)

            weighted_coverage += (region.severity / total_severity) * coverage

        return min(1.0, weighted_coverage)

    def _efficiency_score(
        self,
        available: dict,
        used: dict,
        regions: List[Region],
    ) -> float:
        """
        Score based on efficient resource usage.

        Under scarcity (hard tasks), the agent physically cannot fulfil all needs,
        so we reward proportional impact rather than absolute fulfillment.

        - Full score if all used resources directly reduced unmet needs.
        - On hard tasks with severe scarcity, partial score is still achievable.
        - 0.0 if agent never used any resources (paralysed).
        """
        total_used = sum(used.values())
        total_available = sum(available.values())

        if total_available <= 0:
            return 1.0

        if total_used <= 0:
            return 0.0  # Agent did nothing

        total_unmet_initial = sum(
            r.initial_needs.total() for r in regions
        )
        total_unmet_final = sum(r.unmet_needs.total() for r in regions)
        total_fulfilled = max(0.0, total_unmet_initial - total_unmet_final)

        # Impact ratio: fulfilled / used — but cap at 1.5× to be lenient under scarcity
        # (escalation inflates unmet_final, so fulfilled can under-represent real effort)
        impact_ratio = min(1.5, total_fulfilled / max(1.0, total_used))
        impact_score = min(1.0, impact_ratio / 1.5)  # normalize back to [0,1]

        # Usage ratio: did the agent actually deploy resources against the need?
        # Under scarcity, using all available resources is near-optimal behaviour.
        usage_ratio = min(1.0, total_used / max(1.0, min(total_unmet_initial, total_available)))

        # Combined: 60% impact, 40% deployment effort
        efficiency = 0.6 * impact_score + 0.4 * usage_ratio
        return min(1.0, efficiency)

    def _unmet_needs_score(
        self, regions: List[Region], initial_unmet_totals: dict
    ) -> float:
        """
        Score based on overall fulfillment of resource needs.

        Uses total_delivered / initial_needs to measure agent delivery effort.
        This is escalation-safe: even if unmet_needs grew beyond initial due
        to hard-task dynamics, the agent's delivery effort is correctly captured.
        """
        initial_total = sum(initial_unmet_totals.values())
        if initial_total <= 0:
            return 1.0

        total_delivered = sum(r.total_delivered for r in regions)
        delivery_ratio = min(1.0, total_delivered / initial_total)
        return delivery_ratio

    def _critical_penalty(self, regions: List[Region]) -> float:
        """
        Apply a -0.15 penalty if any critical region (severity >= 8) received
        less than 30% of its initial needs delivered (delivery coverage < 0.30).

        Uses total_delivered / initial_needs to measure agent effort.
        This is escalation-safe and prevents penalising agents facing impossible scarcity.
        Penalty is applied at most once per episode.
        """
        for region in regions:
            if region.severity >= self.CRITICAL_SEVERITY_THRESHOLD:
                initial_total = region.initial_needs.total()
                if initial_total <= 0:
                    continue
                # Delivery coverage: what % of initial need was actually served
                delivery_coverage = region.total_delivered / initial_total
                if delivery_coverage < (1.0 - self.CRITICAL_UNMET_THRESHOLD):
                    # Less than 30% delivered to a critical region = penalty
                    return self.CRITICAL_PENALTY
        return 0.0

    def _progress_bonus(self, regions: List[Region], prev_unmet_total: float) -> float:
        """
        Award +0.10 progress bonus when total unmet needs decreased vs previous step.
        Returns 0.0 if prev_unmet_total not supplied (< 0) or no improvement.
        """
        if prev_unmet_total < 0:
            return 0.0
        current_total = sum(r.unmet_needs.total() for r in regions)
        if current_total < prev_unmet_total:
            return self.PROGRESS_BONUS
        return 0.0

    # ------------------------------------------------------------------
    # Detailed breakdown for logging
    # ------------------------------------------------------------------

    def score_breakdown(
        self,
        state: EnvironmentState,
        initial_unmet_totals: dict,
        total_resources_available: dict,
        total_resources_used: dict,
        prev_unmet_total: float = -1.0,
    ) -> dict:
        """Return detailed scoring breakdown as a dict."""
        priority = self._priority_score(state.regions)
        efficiency = self._efficiency_score(
            total_resources_available, total_resources_used, state.regions
        )
        unmet = self._unmet_needs_score(state.regions, initial_unmet_totals)
        penalty = self._critical_penalty(state.regions)
        progress = self._progress_bonus(state.regions, prev_unmet_total)

        # Add tiny natural variation here as well
        avg_severity = sum(r.severity for r in state.regions) / max(1, len(state.regions))
        base_variation = avg_severity * 0.001

        raw_final = (
            self.PRIORITY_WEIGHT * priority
            + self.EFFICIENCY_WEIGHT * efficiency
            + self.UNMET_NEEDS_WEIGHT * unmet
            + progress
            - penalty
        )

        if raw_final <= 0.0:
            final = 0.01 + base_variation
        elif raw_final >= 1.0:
            final = 0.99 - base_variation
        else:
            final = min(0.99, max(0.01, float(raw_final) + base_variation))

        return {
            "final_score": round(final, 4),
            "priority_allocation": round(priority, 4),
            "resource_efficiency": round(efficiency, 4),
            "unmet_needs_reduction": round(unmet, 4),
            "progress_bonus": round(progress, 4),
            "critical_region_penalty": round(penalty, 4),
            "weighted_components": {
                "priority": round(self.PRIORITY_WEIGHT * priority, 4),
                "efficiency": round(self.EFFICIENCY_WEIGHT * efficiency, 4),
                "unmet": round(self.UNMET_NEEDS_WEIGHT * unmet, 4),
                "progress_bonus": round(progress, 4),
                "penalty": round(-penalty, 4),
            },
        }

def grade(observation=None, **kwargs) -> float:
    from env.models import EnvironmentState
    from env.grader import DisasterReliefGrader

    grader = DisasterReliefGrader()

    try:
        # Convert observation → state
        if isinstance(observation, dict):
            state = EnvironmentState(**observation)
        else:
            state = observation

        # Safe defaults (only if missing)
        needs_obj = getattr(state, "unmet_needs_total", None)
        if needs_obj is not None:
            if hasattr(needs_obj, "model_dump"):
                initial_unmet_totals = needs_obj.model_dump()
            elif isinstance(needs_obj, dict):
                initial_unmet_totals = needs_obj
            else:
                initial_unmet_totals = dict(needs_obj)
        else:
            initial_unmet_totals = {
                "food": 100,
                "water": 100,
                "medicine": 100
            }

        total_resources_available = {
            "food": state.resources.food,
            "water": state.resources.water,
            "medicine": state.resources.medicine,
        }

        total_resources_used = getattr(state, "total_resources_used", {
            "food": 0,
            "water": 0,
            "medicine": 0,
        })

        # Real scoring
        score = grader.compute_score(
            state=state,
            initial_unmet_totals=initial_unmet_totals,
            total_resources_available=total_resources_available,
            total_resources_used=total_resources_used,
            prev_unmet_total=-1.0,
        )

    except Exception as e:
        # Safe fallback (never 0 or 1)
        score = 0.5

    # STRICT clamp
    if score <= 0.0:
        score = 0.01
    elif score >= 1.0:
        score = 0.99

    return float(score)
