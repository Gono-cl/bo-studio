import unittest

import pandas as pd

from core.utils.analysis_utils import (
    ANALYSIS_ORDER_COLUMN,
    infer_db_analysis_context,
    infer_objective_columns,
    prepare_multiobjective_frame,
    prepare_objective_progress_frame,
)


class AnalysisUtilityTests(unittest.TestCase):
    def test_infer_objective_columns_excludes_variables_and_timestamp(self) -> None:
        df = pd.DataFrame(
            {
                "Temperature": [60.0, 80.0],
                "Catalyst": [0.1, 0.2],
                "Yield": [30.0, 45.0],
                "Conversion": [0.7, 0.8],
                "Timestamp": ["2026-08-01 10:00:00", "2026-08-01 11:00:00"],
            }
        )
        variables = [
            ("Temperature", 20.0, 120.0, "C", "continuous"),
            ("Catalyst", 0.0, 1.0, "fraction", "continuous"),
        ]

        self.assertEqual(infer_objective_columns(df, variables), ["Yield", "Conversion"])

    def test_prepare_objective_progress_frame_preserves_original_experiment_numbers(self) -> None:
        df = pd.DataFrame({"Yield": [10.0, None, 30.0]})

        progress = prepare_objective_progress_frame(df, "Yield")

        self.assertEqual(progress[ANALYSIS_ORDER_COLUMN].tolist(), [1, 3])
        self.assertEqual(progress["Yield"].tolist(), [10.0, 30.0])

    def test_prepare_multiobjective_frame_drops_invalid_rows_but_keeps_order(self) -> None:
        df = pd.DataFrame(
            {
                "Yield": [10.0, "bad", 30.0],
                "Conversion": [0.8, 0.9, 0.95],
            }
        )

        cleaned, objectives = prepare_multiobjective_frame(df, ["Yield", "Conversion"])

        self.assertEqual(objectives, ["Yield", "Conversion"])
        self.assertEqual(cleaned[ANALYSIS_ORDER_COLUMN].tolist(), [1, 3])
        self.assertEqual(cleaned["Yield"].tolist(), [10.0, 30.0])

    def test_infer_db_analysis_context_detects_multiobjective_records(self) -> None:
        df = pd.DataFrame(
            {
                "Temperature": [60.0, 80.0],
                "Yield": [30.0, 45.0],
                "Purity": [88.0, 91.0],
            }
        )
        exp = {
            "df_results": df,
            "variables": [("Temperature", 20.0, 120.0, "C", "continuous")],
            "best_result": [
                {"Temperature": 80.0, "Yield": 45.0, "Purity": 91.0},
            ],
            "settings": {
                "objectives": ["Yield", "Purity"],
                "mo_directions": {"Yield": "Maximize", "Purity": "Maximize"},
            },
        }

        context = infer_db_analysis_context(exp)

        self.assertEqual(context["mode"], "mo")
        self.assertEqual(context["mo_objectives"], ["Yield", "Purity"])
        self.assertEqual(context["mo_directions"]["Purity"], "Maximize")

    def test_infer_db_analysis_context_falls_back_to_first_objective_column_for_so(self) -> None:
        df = pd.DataFrame(
            {
                "Temperature": [60.0, 80.0],
                "Yield": [30.0, 45.0],
                "Timestamp": ["2026-08-01 10:00:00", "2026-08-01 11:00:00"],
            }
        )
        exp = {
            "df_results": df,
            "variables": [("Temperature", 20.0, 120.0, "C", "continuous")],
            "best_result": {"Temperature": 80.0, "Yield": 45.0},
            "settings": {},
        }

        context = infer_db_analysis_context(exp)

        self.assertEqual(context["mode"], "so")
        self.assertEqual(context["response"], "Yield")


if __name__ == "__main__":
    unittest.main()
