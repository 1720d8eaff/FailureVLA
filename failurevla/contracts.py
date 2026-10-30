"""Shared in-process policy/backend contracts; timestamps use one monotonic clock."""
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class RobotObservation:
    timestamp: float
    joint_positions: list
    rgb: Any
    depth: Any
    gripper_state: Any


@dataclass(frozen=True)
class GraspCandidate:
    pose: Any
    score: float
    frame_id: str


@dataclass(frozen=True)
class ActionChunk:
    actions: Any
    observation_timestamp: float
    chunk_id: str
    policy_name: str


@dataclass(frozen=True)
class RecoveryRequest:
    failure_type: str
    observation: RobotObservation
    remaining_budget: int
    task_instruction: Optional[str] = None
