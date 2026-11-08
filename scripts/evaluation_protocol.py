"""Large-run interface: retain pilot behavior, align C to the L3 tip field.

The paper does not disclose its view aggregation or prediction-to-tip converter.
This deterministic adapter preserves the model predictions without adding facts.
"""
import json
from pathlib import Path

from protocol import (CONDITIONS as PILOT_CONDITIONS, embguard_jobs, gate_decision, guard_assessments,
                      homeguard_format_audit, homeguard_jobs, inference_settings,
                      load_module, parse_plan, planning_input)
from protocol import planner_prompt as pilot_planner_prompt

CONDITIONS = {name: dict(value) for name, value in PILOT_CONDITIONS.items()}
CONDITIONS["C"].update(prompt="v3_generated_tips", name="homeguard_l3_guidance")


def generated_tips(assessments):
    if assessments is None or len(assessments) != 5:
        raise ValueError("C requires five completed HomeGuard predictions.")
    tips = []
    for view, assessment in enumerate(assessments):
        if assessment.get("view") != view:
            raise ValueError("HomeGuard predictions must retain their view indices.")
        prediction = {key: assessment[key] for key in
                      ("safe", "safety_hazard", "target_object", "constraint_object")}
        tips.append(f"View {view}: " + json.dumps(prediction, ensure_ascii=False))
    return json.dumps(tips, ensure_ascii=False)


def planner_prompt(repo, task, history, condition, safety_notes=None, rejected=None):
    if condition != "C":
        return pilot_planner_prompt(repo, task, history, condition, safety_notes, rejected)
    if rejected:
        raise ValueError("C is advisory guidance and has no action-rejection loop.")
    prompts = load_module(Path(repo) / "og_ego_prim/utils/prompts.py", "isbench_large_prompts")
    fields = planning_input(task)
    fields["history_actions"] = "None" if not history else "\n".join(history)
    fields["safety_tips"] = generated_tips(safety_notes)
    return prompts.V3StepPlanningPrompt.format(**fields)
