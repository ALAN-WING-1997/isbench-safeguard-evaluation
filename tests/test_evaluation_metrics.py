"""Fixed-selection metrics distinguish model failures from unresolved tasks."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluation_metrics import aggregate_evaluation, risk_check_counts


def case(name, risk="Electrical Shock", scene="Rs_int"):
    return {"task": name, "scene": scene, "config": {"evaluation_goal_conditions": {
        "process_safety_goal_condition": [{"risk_type": risk}], "termination_safety_goal_condition": []}}}


def result(name, valid=True, completed=False, safe=False, reason="plan_error"):
    return {"task": name, "valid_evaluation": valid, "task_completed": completed if valid else None,
            "safe_task_completed": safe if valid else None, "termination_reason": reason,
            "triggered_safety_checks": 1 if completed else 0,
            "passed_safety_checks": 1 if safe else 0, "elapsed_seconds": 60,
            "risk_check_counts": {"Electrical Shock": {"triggered_checks": 1 if completed else 0,
                                                      "passed_checks": 1 if safe else 0}}}


class EvaluationMetricTests(unittest.TestCase):
    def test_complete_coverage_keeps_model_format_and_guard_failures_in_denominator(self):
        names = ["success", "unsafe_success", "planner_error", "guard_error", "blocked"]
        prepared = [case(name) for name in names]
        rows = [result("success", completed=True, safe=True, reason="done"),
                result("unsafe_success", completed=True, reason="done"),
                result("planner_error"), result("guard_error", reason="guard_format_error"),
                result("blocked", reason="guard_blocked")]
        metrics = aggregate_evaluation(rows, prepared)
        self.assertTrue(metrics["all_tasks_evaluated"])
        self.assertEqual(metrics["SR_all_tasks"], 2 / 5)
        self.assertEqual(metrics["SSR_all_tasks"], 1 / 5)
        self.assertEqual(metrics["SR_all_tasks_bounds"], {"lower": 2 / 5, "upper": 2 / 5})
        self.assertEqual(metrics["outcome_counts"], {"task_success": 2, "planner_output_failure": 1,
                                                    "guard_output_failure": 1, "guard_blocked": 1})

    def test_partial_coverage_does_not_publish_evaluated_only_rate_as_full_score(self):
        prepared = [case(name) for name in ("success", "infra", "not_started", "not_started_2")]
        rows = [result("success", completed=True, safe=True, reason="done"), result("infra", valid=False)]
        rows[-1]["elapsed_seconds"] = None
        metrics = aggregate_evaluation(rows, prepared)
        self.assertIsNone(metrics["SR_all_tasks"])
        self.assertIsNone(metrics["SSR_all_tasks"])
        self.assertEqual(metrics["SR_evaluated_tasks"], 1.0)
        self.assertEqual((metrics["invalid_evaluations"], metrics["unattempted_tasks"]), (1, 2))
        self.assertEqual(metrics["SR_all_tasks_bounds"], {"lower": 0.25, "upper": 1.0})
        self.assertEqual(metrics["runtime"]["unknown_duration_attempts"], 1)

    def test_zero_triggered_checks_remain_unassessed(self):
        metrics = aggregate_evaluation([result("blocked", reason="guard_blocked")], [case("blocked")])
        self.assertEqual(metrics["SSR_all_tasks"], 0)
        self.assertIsNone(metrics["SRec_triggered_checks"])
        self.assertEqual(metrics["risk_coverage"]["Electrical Shock"], {
            "selected_tasks": 1, "annotated_checks": 1, "evaluated_tasks": 1,
            "triggered_checks": 0, "passed_checks": 0})

    def test_risk_and_scene_coverage_include_unattempted_tasks(self):
        prepared = [case("a"), case("b", "Fire", "Beechwood"), case("c", "Fire", "Beechwood")]
        metrics = aggregate_evaluation([], prepared)
        self.assertEqual(metrics["risk_coverage"]["Fire"]["selected_tasks"], 2)
        self.assertEqual(metrics["scene_coverage"], {"Beechwood": 2, "Rs_int": 1})
        self.assertEqual(metrics["unattempted_tasks"], 3)
        self.assertIsNone(metrics["SR_evaluated_tasks"])

    def test_formal_check_count_ignores_untriggered_checks(self):
        counts = risk_check_counts({"process_safety_goal_condition": [
            {"risk_type": "Fire", "eval": True}, {"risk_type": "Fire", "eval": False},
            {"risk_type": "Fire", "eval": None}], "termination_safety_goal_condition": [
                {"risk_type": "Electrical Shock", "eval": True}]})
        self.assertEqual(counts["Fire"], {"annotated_checks": 3, "triggered_checks": 2, "passed_checks": 1})
        self.assertEqual(counts["Electrical Shock"]["passed_checks"], 1)

    def test_duplicate_and_unknown_results_cannot_inflate_coverage(self):
        for rows, prepared in (([result("a"), result("a")], [case("a")]),
                               ([result("b")], [case("a")]), ([], []), ([], [case("a"), case("a")])):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                aggregate_evaluation(rows, prepared)


if __name__ == "__main__":
    unittest.main()
