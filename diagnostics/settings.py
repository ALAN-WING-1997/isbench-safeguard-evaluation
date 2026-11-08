"""Generation variants for an inference-only diagnostic; no trial defaults change."""
import math

VARIANTS = {
    'sample_8192': {'max_new_tokens': 8192, 'do_sample': True, 'temperature': 1.0, 'top_p': 0.95, 'top_k': 20},
    'greedy_8192': {'max_new_tokens': 8192, 'do_sample': False},
    'sample_2048': {'max_new_tokens': 2048, 'do_sample': True, 'temperature': 1.0, 'top_p': 0.95, 'top_k': 20},
}


def generation_kwargs(request):
    generation = request['generation']
    if type(generation.get('do_sample')) is not bool:
        raise ValueError('do_sample must be a Boolean.')
    tokens = request['settings']['max_new_tokens']
    if type(tokens) is not int or not 1 <= tokens <= 32768:
        raise ValueError('Invalid diagnostic output budget.')
    kwargs = {'max_new_tokens': tokens, 'do_sample': generation['do_sample'],
              'repetition_penalty': 1.0, 'use_cache': True}
    if generation['do_sample']:
        temperature, top_p, top_k = (generation[key] for key in ('temperature', 'top_p', 'top_k'))
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Sampling temperature must be positive and finite.')
        if type(top_p) not in (int, float) or not 0 < top_p <= 1:
            raise ValueError('top_p must be in (0, 1].')
        if type(top_k) is not int or top_k < 0:
            raise ValueError('top_k must be a nonnegative integer.')
        kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)
    return kwargs
