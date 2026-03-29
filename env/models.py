"""
Typed Pydantic models for the AI Disaster Relief Logistics environment.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DisasterType(str, Enum):
    FLOOD = "flood"
    EARTHQUAKE = "earthquake"
    CYCLONE = "cyclone"
    DROUGHT = "drought"


class ResourceType(str, Enum):
    FOOD = "food"
    WATER = "water"
    MEDICINE = "medicine"


# ---------------------------------------------------------------------------
# Resource models
# ---------------------------------------------------------------------------

class ResourcePool(BaseModel):
    """Global supply depot available to the agent."""

    food: float = Field(..., ge=0, description="Food supply in tons")
    water: float = Field(..., ge=0, description="Water supply in kiloliters")
    medicine: float = Field(..., ge=0, description="Medicine units available")
    trucks: int = Field(..., ge=0, description="Transport vehicles")
    fuel: float = Field(..., ge=0, description="Fuel in liters")

    def deduct(self, resource: ResourceType, amount: float) -> None:
        """Deduct an amount from the pool in-place."""
        current = getattr(self, resource.value)
        setattr(self, resource.value, max(0.0, current - amount))

    def available(self, resource: ResourceType) -> float:
        """Return currently available quantity of a resource."""
        return getattr(self, resource.value)


class RegionNeeds(BaseModel):
    """Per-region resource requirements."""

    food: float = Field(default=0.0, ge=0)
    water: float = Field(default=0.0, ge=0)
    medicine: float = Field(default=0.0, ge=0)

    def total(self) -> float:
        return self.food + self.water + self.medicine

    def reduce(self, resource: ResourceType, amount: float) -> float:
        """Reduce a need by amount; returns actual amount fulfilled."""
        current = getattr(self, resource.value)
        fulfilled = min(current, amount)
        setattr(self, resource.value, max(0.0, current - amount))
        return fulfilled


# ---------------------------------------------------------------------------
# Region model
# ---------------------------------------------------------------------------

class Region(BaseModel):
    """A single affected geographic region."""

    id: str = Field(..., description="Unique region identifier")
    name: str = Field(..., description="Human-readable region name")
    severity: int = Field(..., ge=1, le=10, description="Disaster severity 1-10")
    population: int = Field(..., ge=0, description="Affected population")
    disaster_type: DisasterType
    location: str = Field(..., description="Geographic location description")
    needs: RegionNeeds = Field(default_factory=RegionNeeds)
    unmet_needs: RegionNeeds = Field(default_factory=RegionNeeds)
    initial_needs: RegionNeeds = Field(default_factory=RegionNeeds)
    deaths_this_step: int = Field(default=0, ge=0)
    total_deaths: int = Field(default=0, ge=0)
    total_delivered: float = Field(default=0.0, ge=0, description="Cumulative resources delivered (unaffected by escalation)")

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: int) -> int:
        if not 1 <= v <= 10:
            raise ValueError(f"Severity must be between 1 and 10, got {v}")
        return v

    def apply_delivery(self, resource: ResourceType, amount: float) -> float:
        """
        Apply a resource delivery to this region.
        Returns the amount of unmet need actually fulfilled.
        Also increments total_delivered to track cumulative delivery effort.
        """
        fulfilled = self.unmet_needs.reduce(resource, amount)
        self.total_delivered += fulfilled
        return fulfilled

    def compute_deaths(self) -> int:
        """Estimate deaths from persistent unmet needs based on severity."""
        severity_factor = self.severity / 10.0
        unmet_ratio = (
            self.unmet_needs.total() / max(1.0, self.initial_needs.total())
        )
        deaths = int(
            self.population * unmet_ratio * severity_factor * 0.001
        )
        return deaths

    def escalate_needs(self, factor: float = 1.1) -> None:
        """Increase unmet needs over time (dynamic deterioration for hard tasks)."""
        self.unmet_needs.food = min(
            self.unmet_needs.food * factor, self.needs.food * 2
        )
        self.unmet_needs.water = min(
            self.unmet_needs.water * factor, self.needs.water * 2
        )
        self.unmet_needs.medicine = min(
            self.unmet_needs.medicine * factor, self.needs.medicine * 2
        )


# ---------------------------------------------------------------------------
# Action models
# ---------------------------------------------------------------------------

class Delivery(BaseModel):
    """A single resource delivery from the depot to a region."""

    region_id: str = Field(..., description="Target region ID")
    resource: ResourceType = Field(..., description="Resource type to deliver")
    amount: float = Field(..., ge=0, description="Quantity to deliver")

    @field_validator("amount")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Delivery amount must be non-negative")
        return v


class AgentAction(BaseModel):
    """Full action submitted by the agent in one step."""

    deliveries: List[Delivery] = Field(
        default_factory=list,
        description="List of delivery operations",
    )


# ---------------------------------------------------------------------------
# State model
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    """Complete snapshot of the environment at a given time step."""

    time_step: int = Field(default=0, ge=0)
    max_steps: int = Field(..., ge=1)
    resources: ResourcePool
    regions: List[Region]
    unmet_needs_total: RegionNeeds = Field(default_factory=RegionNeeds)
    deaths_averted: int = Field(default=0, ge=0)
    episode_score: float = Field(default=0.0, ge=0.0, le=1.0)
    task_id: str = Field(default="easy")
    done: bool = Field(default=False)

    def recompute_unmet_totals(self) -> None:
        """Aggregate unmet needs across all regions."""
        self.unmet_needs_total = RegionNeeds(
            food=sum(r.unmet_needs.food for r in self.regions),
            water=sum(r.unmet_needs.water for r in self.regions),
            medicine=sum(r.unmet_needs.medicine for r in self.regions),
        )


# ---------------------------------------------------------------------------
# API response models
# ---------------------------------------------------------------------------

class ResetResponse(BaseModel):
    """Response returned by /reset endpoint."""

    state: EnvironmentState
    message: str = "Environment reset successfully"


class StepResponse(BaseModel):
    """Response returned by /step endpoint."""

    reward: float = Field(..., description="Step reward")
    done: bool = Field(..., description="Whether the episode has ended")
    info: Dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "1.0.0"
    environment: str = "ai-disaster-relief-logistics"
