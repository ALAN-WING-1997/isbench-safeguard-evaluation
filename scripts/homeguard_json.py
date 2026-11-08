"""Constrain HomeGuard's final JSON without masking its reasoning tokens."""
from protocol import inspect_homeguard


HOMEGUARD_JSON_FORMAT = "homeguard_json_v1"
THINK_END = "</think>"
BOX_LIST = {
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
}
HOMEGUARD_SCHEMA = {
    "type": "object",
    "properties": {
        "safe": {"type": "boolean"},
        "safety_hazard": {"anyOf": [{"type": "null"}, {"type": "string", "minLength": 1}]},
        "target_object": BOX_LIST,
        "constraint_object": BOX_LIST,
    },
    "required": ["safe", "safety_hazard", "target_object", "constraint_object"],
    "additionalProperties": False,
}


class HomeGuardJSONDecoder:
    """Reuse one tokenizer trie across views and create a fresh parser for each view."""

    def __init__(self, tokenizer):
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data

        delimiter = tokenizer.encode(THINK_END, add_special_tokens=False)
        if len(delimiter) != 1 or tokenizer.decode(delimiter, skip_special_tokens=True) != THINK_END:
            raise ValueError("HomeGuard must have a visible, single-token reasoning delimiter.")
        if tokenizer.eos_token_id is None:
            raise ValueError("HomeGuard tokenizer is missing its end-of-generation token.")
        self.delimiter_id = delimiter[0]
        self.eos_token_id = tokenizer.eos_token_id
        self.tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)

    def constraint(self, input_tokens):
        from lmformatenforcer import JsonSchemaParser, TokenEnforcer

        return HomeGuardJSONConstraint(
            TokenEnforcer(self.tokenizer_data, JsonSchemaParser(HOMEGUARD_SCHEMA)),
            input_tokens,
            self.delimiter_id,
        )


class HomeGuardJSONConstraint:
    """Transformers logits processor for one sequence in one generate call."""

    def __init__(self, enforcer, input_tokens, delimiter_id):
        self.enforcer = enforcer
        self.input_tokens = input_tokens
        self.delimiter_id = delimiter_id
        self.final_start = None

    def __call__(self, input_ids, scores):
        token_count = input_ids.shape[1]
        if self.final_start is None:
            if token_count <= self.input_tokens or input_ids[0, -1].item() != self.delimiter_id:
                return scores
            self.final_start = token_count

        allowed = self.enforcer.get_allowed_tokens(input_ids[0, self.final_start:].tolist()).allowed_tokens
        # Keep the reasoning delimiter out of strings so later parsing has one boundary.
        allowed = [token for token in allowed if token != self.delimiter_id]
        masked = scores.new_full(scores.shape, -float("inf"))
        masked[:, allowed] = scores[:, allowed]
        return masked

    def format_error(self, raw):
        """Reject missing grammar activation and final schema violations."""
        if self.final_start is None:
            return "HomeGuard did not enter constrained JSON generation."
        _, delimiter, final = raw.partition(THINK_END)
        if not delimiter or not final.lstrip().startswith("{"):
            return "Missing constrained JSON object after the reasoning delimiter."
        if THINK_END in final or "</thinking>" in final:
            return "Unexpected reasoning delimiter in the final assessment."
        audit = inspect_homeguard(final)
        if audit["status"] != "accepted":
            return audit["error"] or "Constrained final answer requires canonical JSON coordinates."
        assessment = audit["assessment"]
        if assessment["safe"] and assessment["safety_hazard"] is not None:
            return "A safe assessment must have a null safety_hazard."
        return None


def preflight_homeguard_json(tokenizer):
    """Check the constrained token paths using the locked tokenizer and no model weights."""
    decoder = HomeGuardJSONDecoder(tokenizer)

    def permits_complete_answer(text):
        constraint = decoder.constraint(input_tokens=0)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        for index, token in enumerate(tokens):
            if token not in constraint.enforcer.get_allowed_tokens(tokens[:index]).allowed_tokens:
                return False
        return decoder.eos_token_id in constraint.enforcer.get_allowed_tokens(tokens).allowed_tokens

    canonical = (
        '{"safe": true, "safety_hazard": null, "target_object": [], "constraint_object": []}',
        '{"safe": false, "safety_hazard": "A hot surface", "target_object": [[1, 2, 3, 4]], '
        '"constraint_object": [[10, 20, 30, 40]]}',
    )
    malformed = (
        canonical[0].replace('true', '"true"'),
        canonical[0].replace(', "constraint_object": []', ''),
        canonical[1].replace('10, 20, 30, 40', '10, 20, 30'),
        canonical[1][:-1] + ', "unexpected": null}',
    )
    if not all(permits_complete_answer(text) for text in canonical):
        raise ValueError("HomeGuard grammar rejected a canonical JSON example.")
    if any(permits_complete_answer(text) for text in malformed):
        raise ValueError("HomeGuard grammar allowed a malformed JSON example.")
    return {"response_format": HOMEGUARD_JSON_FORMAT, "canonical_examples_checked": len(canonical),
            "malformed_examples_checked": len(malformed)}
