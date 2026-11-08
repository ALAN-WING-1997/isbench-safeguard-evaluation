"""Check L3 alignment and isolation without changing pilot behavior."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent / "is-bench-reproduction"
TASK = REPO / "data/tasks/moving_electric_fan_on_sofa.json"
sys.path.insert(0, str(ROOT / "scripts"))
import evaluation_protocol
import protocol


class EvaluationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.notes = [{"view": i, "safe": False, "safety_hazard": f"Predicted hazard {i}",
                       "target_object": [[1, 2, 3, 4]], "constraint_object": []} for i in range(5)]

    @unittest.skipUnless(REPO.exists(), "Pinned IS-Bench checkout is not available in ../is-bench-reproduction.")
    def test_c_uses_upstream_l3_and_only_predicted_tips(self):
        task = json.loads(TASK.read_text())
        task["evaluation_cautions"] = ["SECRET_ANNOTATED_TIP"]
        task["evaluation_goal_conditions"]["process_safety_goal_condition"][0]["safety_principle"] = "SECRET_SAFETY_PREDICATE"
        task["example_planning"] = ["SECRET_REFERENCE_ACTION"]
        prompt = evaluation_protocol.planner_prompt(REPO, task, [], "C", self.notes)
        module = protocol.load_module(REPO / "og_ego_prim/utils/prompts.py", "test_l3_prompts")
        fields = protocol.planning_input(task)
        fields.update(history_actions="None", safety_tips=evaluation_protocol.generated_tips(self.notes))
        self.assertEqual(prompt, module.V3StepPlanningPrompt.format(**fields))
        for secret in ("SECRET_ANNOTATED_TIP", "SECRET_SAFETY_PREDICATE", "SECRET_REFERENCE_ACTION"):
            self.assertNotIn(secret, prompt)
        for i in range(5):
            self.assertIn(f"Predicted hazard {i}", prompt)
        self.assertNotIn("Safety assessments predicted from the current views", prompt)

    @unittest.skipUnless(REPO.exists(), "Pinned IS-Bench checkout is not available in ../is-bench-reproduction.")
    def test_a_b_d_remain_equal_to_pilot_prompts(self):
        task = json.loads(TASK.read_text())
        for condition in "ABD":
            rejected = [{"action": "wipe(x,y)", "assessments": []}] if condition == "D" else []
            self.assertEqual(evaluation_protocol.planner_prompt(REPO, task, ["1. done()"], condition, rejected=rejected),
                             protocol.planner_prompt(REPO, task, ["1. done()"], condition, rejected=rejected))

    def test_invalid_view_list_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluation_protocol.generated_tips(self.notes[:4])
        self.notes[0]["view"] = 3
        with self.assertRaises(ValueError):
            evaluation_protocol.generated_tips(self.notes)

if __name__ == "__main__":
    unittest.main()
