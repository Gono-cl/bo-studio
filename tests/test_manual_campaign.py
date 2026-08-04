import unittest

import pandas as pd

from core.utils.manual_campaign import (
    campaign_has_started,
    parse_required_result_text,
    validate_initial_results,
)


class ManualCampaignStateTests(unittest.TestCase):
    def test_campaign_has_not_started_for_fresh_state(self) -> None:
        self.assertFalse(
            campaign_has_started(
                manual_initialized=False,
                manual_data=[],
                suggestions=[],
                iteration=0,
                next_suggestion_cached=None,
            )
        )

    def test_campaign_has_started_when_initial_suggestions_exist(self) -> None:
        self.assertTrue(
            campaign_has_started(
                manual_initialized=False,
                manual_data=[],
                suggestions=[[42.0]],
                iteration=0,
                next_suggestion_cached=None,
            )
        )

    def test_campaign_has_started_when_data_exists(self) -> None:
        self.assertTrue(
            campaign_has_started(
                manual_initialized=False,
                manual_data=[{"Temperature": 80.0, "Yield": 50.0}],
                suggestions=[],
                iteration=0,
                next_suggestion_cached=None,
            )
        )


class ManualCampaignInputTests(unittest.TestCase):
    def test_parse_required_result_text_accepts_comma_decimal(self) -> None:
        self.assertAlmostEqual(parse_required_result_text("3,5"), 3.5)

    def test_parse_required_result_text_rejects_blank(self) -> None:
        with self.assertRaisesRegex(ValueError, "required"):
            parse_required_result_text("")

    def test_validate_initial_results_returns_ordered_rows(self) -> None:
        df = pd.DataFrame(
            [
                {"Experiment": 1, "Temperature": 60.0, "Catalyst": 0.2, "Yield": "15.5"},
                {"Experiment": 2, "Temperature": 90.0, "Catalyst": 0.7, "Yield": 42},
            ]
        )
        variables = [
            ("Temperature", 20.0, 120.0, "C", "continuous"),
            ("Catalyst", 0.0, 1.0, "fraction", "continuous"),
        ]
        rows = validate_initial_results(df, variables, "Yield")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], [60.0, 0.2])
        self.assertAlmostEqual(rows[0][1], 15.5)
        self.assertAlmostEqual(rows[1][1], 42.0)

    def test_validate_initial_results_rejects_missing_values(self) -> None:
        df = pd.DataFrame(
            [
                {"Experiment": 1, "Temperature": 60.0, "Yield": ""},
                {"Experiment": 2, "Temperature": 90.0, "Yield": 42},
            ]
        )
        variables = [("Temperature", 20.0, 120.0, "C", "continuous")]
        with self.assertRaisesRegex(ValueError, "Missing values"):
            validate_initial_results(df, variables, "Yield")

    def test_validate_initial_results_rejects_invalid_numbers(self) -> None:
        df = pd.DataFrame(
            [
                {"Experiment": 1, "Temperature": 60.0, "Yield": "abc"},
                {"Experiment": 2, "Temperature": 90.0, "Yield": 42},
            ]
        )
        variables = [("Temperature", 20.0, 120.0, "C", "continuous")]
        with self.assertRaisesRegex(ValueError, "not valid numbers"):
            validate_initial_results(df, variables, "Yield")


if __name__ == "__main__":
    unittest.main()
