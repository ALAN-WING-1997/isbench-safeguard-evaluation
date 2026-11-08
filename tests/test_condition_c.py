"""Verify the revised C information boundary and the non-executing stop outcome."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent / "is-bench-reproduction"
sys.path.insert(0, str(ROOT / "scripts"))
from condition_c import compact_tips, improved_homeguard_jobs, improved_planner_prompt, parse_improved_plan
from protocol import parse_plan


class RevisedConditionCTests(unittest.TestCase):
    def setUp(self):
        self.reports = [
            {"view": view, "safe": True, "safety_hazard": None, "target_object": [], "constraint_object": []}
            for view in range(5)
        ]

    def test_grouping_preserves_conflict_and_view_coordinates(self):
        hazard = "Visible exposed wiring on an unlisted appliance beside the filter."
        for view in (0, 3):
            self.reports[view].update(safe=False, safety_hazard=hazard, constraint_object=[[view, 1, 5, 6]])
        self.reports[4].update(safe=False, safety_hazard="Insufficient evidence: state partly obscured.")
        groups = json.loads(compact_tips(self.reports))
        self.assertEqual(groups, [
            {"safe": False, "safety_hazard": hazard, "views": [0, 3], "boxes_by_view": {
                "0": {"constraint_object": [[0, 1, 5, 6]]}, "3": {"constraint_object": [[3, 1, 5, 6]]}}},
            {"safe": True, "safety_hazard": None, "views": [1, 2]},
            {"safe": False, "safety_hazard": "Insufficient evidence: state partly obscured.", "views": [4]},
        ])
        with self.assertRaises(ValueError):
            compact_tips(self.reports[:4])

    @unittest.skipUnless(REPO.exists(), "Pinned IS-Bench checkout is not available in ../is-bench-reproduction.")
    def test_prompts_use_execution_outcomes_without_scoring_annotations(self):
        task = json.loads((REPO / "data/tasks/clean_an_air_filter.json").read_text())
        task["evaluation_cautions"] = ["PRIVATE_SAFETY_CANARY"]
        task["evaluation_goal_conditions"]["process_safety_goal_condition"] = ["PRIVATE_PREDICATE_CANARY"]
        task["example_planning"] = ["REFERENCE_ACTION_CANARY"]
        history = ["1. OPEN(cabinet.n.01_1) failed: blocked by an obstacle."]
        images = [f"view-{view}.png" for view in range(5)]
        jobs = improved_homeguard_jobs(images, task, history)
        self.assertEqual([job["view"] for job in jobs], list(range(5)))
        self.assertEqual([job["images"] for job in jobs], [[image] for image in images])
        self.assertTrue(all(job["response_format"] == "homeguard_json_v1" for job in jobs))
        planner = improved_planner_prompt(REPO, task, history, self.reports)
        self.assertNotIn("ensure that all safety tips are followed", planner)
        for prompt in [job["prompt"] for job in jobs] + [planner]:
            self.assertIn(history[0], prompt)
            for private in ("PRIVATE_SAFETY_CANARY", "PRIVATE_PREDICATE_CANARY", "REFERENCE_ACTION_CANARY"):
                self.assertNotIn(private, prompt)

    def test_blocked_is_a_revised_c_only_outcome_with_a_reason(self):
        raw = '</think>{"action":"BLOCKED()","caution":"Unsafe appliance cannot be switched off."}'
        self.assertIsNone(parse_plan(raw, {}, []))
        self.assertEqual(parse_improved_plan(raw, {}, []), {
            "action": "blocked()", "caution": "Unsafe appliance cannot be switched off.",
        })
        self.assertIsNone(parse_improved_plan('{"action":"BLOCKED()","caution":null}', {}, []))
        self.assertIsNone(parse_improved_plan('{"action":"TOGGLE_OFF(invented_stove)"}', {"TOGGLE_OFF": 1}, []))
