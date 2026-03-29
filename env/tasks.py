"""
Task definitions for the AI Disaster Relief Logistics environment.
Three difficulty levels: EASY, MEDIUM, HARD.
All tasks are deterministic and reproducible.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Dict, List

from env.models import (
    DisasterType,
    Region,
    RegionNeeds,
    ResourcePool,
)


# ---------------------------------------------------------------------------
# Base Task
# ---------------------------------------------------------------------------

class BaseTask(ABC):
    """Abstract base class for all environment tasks."""

    task_id: str = "base"
    difficulty: str = "base"
    max_steps: int = 5
    escalate_needs: bool = False
    escalation_factor: float = 1.0

    @abstractmethod
    def get_initial_resources(self) -> ResourcePool:
        ...

    @abstractmethod
    def get_regions(self) -> List[Region]:
        ...

    def get_config(self) -> Dict:
        return {
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "max_steps": self.max_steps,
            "escalate_needs": self.escalate_needs,
            "escalation_factor": self.escalation_factor,
        }


# ---------------------------------------------------------------------------
# EASY Task — 2 Regions, Sufficient Resources
# ---------------------------------------------------------------------------

class EasyTask(BaseTask):
    """
    Flood scenario with 2 regions and fully sufficient resources.
    Teaches basic severity-based prioritization.
    """

    task_id = "easy"
    difficulty = "easy"
    max_steps = 5
    escalate_needs = False

    def get_initial_resources(self) -> ResourcePool:
        return ResourcePool(
            food=500.0,
            water=800.0,
            medicine=300.0,
            trucks=10,
            fuel=2000.0,
        )

    def get_regions(self) -> List[Region]:
        r1_needs = RegionNeeds(food=100.0, water=150.0, medicine=50.0)
        r2_needs = RegionNeeds(food=60.0, water=80.0, medicine=30.0)

        return [
            Region(
                id="R1",
                name="Riverside District",
                severity=8,
                population=25000,
                disaster_type=DisasterType.FLOOD,
                location="Northern floodplain, 40km from relief depot",
                needs=copy.deepcopy(r1_needs),
                unmet_needs=copy.deepcopy(r1_needs),
                initial_needs=copy.deepcopy(r1_needs),
            ),
            Region(
                id="R2",
                name="Coastal Village",
                severity=4,
                population=8000,
                disaster_type=DisasterType.FLOOD,
                location="Southern coastline, 15km from relief depot",
                needs=copy.deepcopy(r2_needs),
                unmet_needs=copy.deepcopy(r2_needs),
                initial_needs=copy.deepcopy(r2_needs),
            ),
        ]


# ---------------------------------------------------------------------------
# MEDIUM Task — 4 Regions, Limited Resources
# ---------------------------------------------------------------------------

class MediumTask(BaseTask):
    """
    Earthquake scenario with 4 regions and limited supplies.
    Agent must triage and allocate scarce resources wisely.
    """

    task_id = "medium"
    difficulty = "medium"
    max_steps = 8
    escalate_needs = False

    def get_initial_resources(self) -> ResourcePool:
        return ResourcePool(
            food=300.0,
            water=400.0,
            medicine=150.0,
            trucks=8,
            fuel=1500.0,
        )

    def get_regions(self) -> List[Region]:
        specs = [
            ("R1", "City Center", 9, 80000, DisasterType.EARTHQUAKE,
             "Urban core, infrastructure destroyed",
             RegionNeeds(food=120.0, water=200.0, medicine=80.0)),
            ("R2", "Industrial Zone", 6, 15000, DisasterType.EARTHQUAKE,
             "Factory district, partial collapse",
             RegionNeeds(food=60.0, water=90.0, medicine=35.0)),
            ("R3", "Mountain Village", 7, 12000, DisasterType.EARTHQUAKE,
             "Remote hilltop, road access cut off",
             RegionNeeds(food=80.0, water=100.0, medicine=50.0)),
            ("R4", "Suburban Area", 3, 40000, DisasterType.EARTHQUAKE,
             "Residential zone, moderate damage",
             RegionNeeds(food=40.0, water=60.0, medicine=20.0)),
        ]

        regions = []
        for rid, name, sev, pop, dtype, loc, needs in specs:
            regions.append(
                Region(
                    id=rid,
                    name=name,
                    severity=sev,
                    population=pop,
                    disaster_type=dtype,
                    location=loc,
                    needs=copy.deepcopy(needs),
                    unmet_needs=copy.deepcopy(needs),
                    initial_needs=copy.deepcopy(needs),
                )
            )
        return regions


# ---------------------------------------------------------------------------
# HARD Task — 6 Regions, Severe Scarcity + Dynamic Deterioration
# ---------------------------------------------------------------------------

class HardTask(BaseTask):
    """
    Cyclone scenario with 6 regions, severe scarcity, and escalating needs.
    Unmet needs grow each step — agent must adapt dynamically.
    """

    task_id = "hard"
    difficulty = "hard"
    max_steps = 12
    escalate_needs = True
    escalation_factor = 1.08  # 8% increase per step

    def get_initial_resources(self) -> ResourcePool:
        return ResourcePool(
            food=250.0,
            water=300.0,
            medicine=100.0,
            trucks=6,
            fuel=1000.0,
        )

    def get_regions(self) -> List[Region]:
        specs = [
            ("R1", "Port City", 10, 120000, DisasterType.CYCLONE,
             "Major port, catastrophic damage, flooding ongoing",
             RegionNeeds(food=100.0, water=180.0, medicine=70.0)),
            ("R2", "Fishing Harbor", 8, 30000, DisasterType.CYCLONE,
             "Harbor destroyed, stranded population",
             RegionNeeds(food=70.0, water=110.0, medicine=45.0)),
            ("R3", "Agricultural Plains", 6, 50000, DisasterType.CYCLONE,
             "Crops destroyed, livestock lost",
             RegionNeeds(food=90.0, water=80.0, medicine=30.0)),
            ("R4", "Hill Station", 5, 18000, DisasterType.CYCLONE,
             "Landslide risk, road blocked",
             RegionNeeds(food=50.0, water=60.0, medicine=25.0)),
            ("R5", "Industrial Corridor", 7, 45000, DisasterType.CYCLONE,
             "Chemical plant damage, contamination risk",
             RegionNeeds(food=60.0, water=90.0, medicine=55.0)),
            ("R6", "Remote Island", 9, 20000, DisasterType.CYCLONE,
             "Completely isolated, aerial drop needed",
             RegionNeeds(food=80.0, water=100.0, medicine=60.0)),
        ]

        regions = []
        for rid, name, sev, pop, dtype, loc, needs in specs:
            regions.append(
                Region(
                    id=rid,
                    name=name,
                    severity=sev,
                    population=pop,
                    disaster_type=dtype,
                    location=loc,
                    needs=copy.deepcopy(needs),
                    unmet_needs=copy.deepcopy(needs),
                    initial_needs=copy.deepcopy(needs),
                )
            )
        return regions


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, type[BaseTask]] = {
    "easy": EasyTask,
    "medium": MediumTask,
    "hard": HardTask,
}


def get_task(task_id: str) -> BaseTask:
    """Instantiate a task by ID."""
    if task_id not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{task_id}'. Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_id]()
