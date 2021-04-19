"""

"""
from src.probability.bayes_theorem import BayesTheorem


class TestBayesTheorem:
    def test_calculate(self):
        instance = BayesTheorem(
            probability_hypothesis=1 / 21,
            probability_evidence_given_hypothesis=0.4,
            probability_not_hypothesis=20 / 21,
            probability_evidence_given_not_hypothesis=0.1,
        )
        result = 4 / 24
        assert round(instance.calculate(), 4) == round(result, 4)
