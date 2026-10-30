"""Pluggable recovery decisions, bounded per episode."""
from dataclasses import dataclass


@dataclass
class RecoveryController:
    mode: str
    budget: int
    threshold: float = 0.7
    learned_selector: object = None

    def __post_init__(self):
        if self.mode not in {'none', 'fixed', 'rule', 'learned'} or self.budget < 0 or not 0 <= self.threshold <= 1:
            raise ValueError('invalid recovery configuration')
        if self.mode == 'learned' and self.learned_selector is None:
            raise ValueError('learned mode needs a selector')

    def decide(self, failure_type, probability):
        if not 0 <= probability <= 1:
            raise ValueError('invalid probability')
        if self.budget == 0 or self.mode == 'none' or failure_type == 'none':
            return 'terminate'
        if self.mode != 'fixed' and probability < self.threshold:
            return 'continue'
        decision = ('retry' if self.mode == 'fixed' else
                    self.learned_selector(failure_type, probability) if self.mode == 'learned' else
                    {'visual_localization': 'reobserve', 'grasp': 'regrasp', 'execution_deviation': 'replan',
                     'target_change': 'reobserve', 'premature_termination': 'replan'}.get(failure_type, 'terminate'))
        if decision not in {'retry', 'reobserve', 'regrasp', 'replan', 'continue', 'terminate'}:
            raise ValueError('invalid recovery decision')
        if decision not in {'continue', 'terminate'}:
            self.budget -= 1
        return decision
