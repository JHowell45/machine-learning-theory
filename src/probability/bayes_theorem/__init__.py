"""

"""
from dataclasses import dataclass


@dataclass
class BayesTheorem:
    probability_hypothesis: float
    probability_evidence_given_hypothesis: float
    probability_not_hypothesis: float
    probability_evidence_given_not_hypothesis: float

    @property
    def probability_evidence(self) -> float:
        return (
            self.probability_hypothesis * self.probability_evidence_given_hypothesis
        ) + (
            self.probability_not_hypothesis
            * self.probability_evidence_given_not_hypothesis
        )

    def calculate(self) -> float:
        return (
            self.probability_hypothesis * self.probability_evidence_given_hypothesis
        ) / self.probability_evidence
