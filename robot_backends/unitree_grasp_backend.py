"""Adapter for an injected unitree-g1d-grasp bridge; no hardware imports."""
from failurevla.contracts import RecoveryRequest


class UnitreeGraspBackend:
    def __init__(self, bridge):
        self.bridge = bridge

    def observe(self):
        return self.bridge.observe()

    def plan_grasp_recovery(self, failure_type, remaining_budget, instruction=None):
        if remaining_budget <= 0:
            return None
        obs = self.observe()
        return self.bridge.plan_recovery(RecoveryRequest(failure_type, obs, remaining_budget, instruction))

    def execute_reviewed_plan(self, plan):
        return self.bridge.execute_reviewed_plan(plan)
