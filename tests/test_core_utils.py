import unittest

import numpy as np

from core.utils.hypervolume import hypervolume_2d
from core.utils.knee import knee_index_2d
from core.utils.n_init_guidance import format_n_init_range, recommend_n_init_range
from core.utils.pareto import is_nondominated, pareto_front_indices


class NInitGuidanceTests(unittest.TestCase):
    def test_format_n_init_range_handles_single_value(self) -> None:
        self.assertEqual(format_n_init_range(6, 6), "6")

    def test_format_n_init_range_handles_range(self) -> None:
        self.assertEqual(format_n_init_range(6, 10), "6-10")

    def test_recommend_n_init_range_expands_for_multiobjective(self) -> None:
        low, high, text = recommend_n_init_range(4, multiobjective=True)
        self.assertEqual((low, high), (12, 20))
        self.assertIn("multiobjective", text)

    def test_recommend_n_init_range_respects_budget_when_possible(self) -> None:
        low, high, text = recommend_n_init_range(2, total_budget=20)
        self.assertEqual((low, high), (4, 6))
        self.assertNotIn("below this rule-of-thumb scale", text)

    def test_recommend_n_init_range_warns_when_budget_is_too_small(self) -> None:
        low, high, text = recommend_n_init_range(5, total_budget=6)
        self.assertEqual((low, high), (10, 15))
        self.assertIn("below this rule-of-thumb scale", text)


class ParetoUtilityTests(unittest.TestCase):
    def test_is_nondominated_mask_for_maximization(self) -> None:
        points = np.array(
            [
                [5.0, 1.0],
                [4.0, 3.0],
                [3.0, 2.0],
                [2.0, 5.0],
            ]
        )
        mask = is_nondominated(points)
        np.testing.assert_array_equal(mask, np.array([True, True, False, True]))

    def test_pareto_front_indices_are_sorted_by_objective_sum(self) -> None:
        points = np.array(
            [
                [1.0, 5.0],
                [5.0, 1.0],
                [4.0, 3.0],
                [3.0, 2.0],
            ]
        )
        self.assertEqual(pareto_front_indices(points), [2, 0, 1])


class TradeoffAnalysisTests(unittest.TestCase):
    def test_hypervolume_2d_uses_reference_rectangle_sum(self) -> None:
        points = np.array(
            [
                [1.0, 4.0],
                [2.0, 3.0],
                [4.0, 1.0],
            ]
        )
        self.assertAlmostEqual(hypervolume_2d(points, ref=(0.0, 0.0)), 9.0)

    def test_knee_index_2d_returns_none_for_short_front(self) -> None:
        points = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.assertIsNone(knee_index_2d(points))

    def test_knee_index_2d_identifies_bend_point(self) -> None:
        points = np.array(
            [
                [0.0, 1.0],
                [0.35, 0.88],
                [0.55, 0.62],
                [1.0, 0.0],
            ]
        )
        self.assertEqual(knee_index_2d(points), 1)


if __name__ == "__main__":
    unittest.main()
