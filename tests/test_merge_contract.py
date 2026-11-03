import json
import tempfile
import unittest
from pathlib import Path

from merge_lora import normalize_non_lora_keys, validate_full_model


class MergeContract(unittest.TestCase):
    def test_official_prefixes_preserve_projector_and_embedding(self):
        state = {'base_model.model.model.mm_projector.weight': 1,
                 'base_model.model.lm_head.weight': 2}
        self.assertEqual(normalize_non_lora_keys(state),
                         {'model.mm_projector.weight': 1, 'lm_head.weight': 2})

    def test_adapter_and_missing_shards_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'config.json').write_text('{}')
            (root/'adapter_model.safetensors').write_bytes(b'x'*2048)
            with self.assertRaises(ValueError):
                validate_full_model(root)
            (root/'model.safetensors.index.json').write_text(json.dumps(
                {'weight_map': {'a':'model-00001-of-00002.safetensors',
                                'b':'model-00002-of-00002.safetensors'}}))
            (root/'model-00001-of-00002.safetensors').write_bytes(b'x'*2048)
            with self.assertRaises(ValueError):
                validate_full_model(root)
            (root/'model-00002-of-00002.safetensors').write_bytes(b'x'*2048)
            validate_full_model(root)
