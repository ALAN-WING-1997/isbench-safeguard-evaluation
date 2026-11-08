"""Regression for EMBGuard's failure in multimodal chat preprocessing."""
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from protocol import embguard_jobs, homeguard_jobs
from vlm_worker import chat_messages
from prepare_models import check_multimodal_chat


class WorkerMessageTests(unittest.TestCase):
    def test_embguard_system_message_survives_processor_visual_scan(self):
        job = embguard_jobs(['view.png'], 'toggle_off(electric_fan.n.01_1)')[0]
        image = object()
        messages = chat_messages(job, [image])
        # This is the exact operation that raised TypeError in the saved traceback.
        visuals = [[content for content in message['content']
                    if content['type'] in ['image', 'video']] for message in messages]
        self.assertEqual(visuals, [[], [{'type': 'image', 'image': image}]])
        self.assertEqual(messages[0], {'role': 'system', 'content': [
            {'type': 'text', 'text': job['system']}]})
        self.assertEqual(messages[1]['content'][-1], {'type': 'text', 'text': job['prompt']})
        self.assertEqual([m['role'] for m in messages], ['system', 'user'])

    def test_jobs_without_system_keep_existing_user_content(self):
        images = [object(), object()]
        for job in ({'images': ['a', 'b'], 'prompt': 'planner prompt'},
                    homeguard_jobs(['a'], 'Move the fan')[0]):
            expected = [{'role': 'user', 'content': [
                *[{'type': 'image', 'image': img} for img in images],
                {'type': 'text', 'text': job['prompt']}]}]
            self.assertEqual(chat_messages(job, images), expected)

    def test_processor_preflight_exercises_system_and_image_without_generation(self):
        image = SimpleNamespace(__enter__=None)
        class FakeImage:
            def __enter__(self):
                return image
            def __exit__(self, *args):
                pass
        pil = ModuleType('PIL')
        pil.Image = SimpleNamespace(new=lambda *a, **kw: FakeImage())
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                visuals = [c for m in messages for c in m['content'] if c['type'] in ['image', 'video']]
                self.assertions = (messages, visuals, kwargs)
                return {'input_ids': SimpleNamespace(shape=(1, 17))}
        processor = Processor()
        with patch.dict(sys.modules, {'PIL': pil}):
            result = check_multimodal_chat(processor)
        messages, visuals, kwargs = processor.assertions
        self.assertEqual(messages[0]['role'], 'system')
        self.assertIs(visuals[0]['image'], image)
        self.assertTrue(kwargs['tokenize'])
        self.assertEqual(result, {'system_text_blocks': True, 'input_tokens': 17, 'model_generation_performed': False})


if __name__ == '__main__':
    unittest.main()
