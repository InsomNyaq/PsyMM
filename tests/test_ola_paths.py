import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('ola_local_paths',
    Path(__file__).resolve().parents[1]/'LLM_needs/Ola-main/ola/model/local_paths.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EncoderMigration(unittest.TestCase):
    def test_migrated_checkpoint_paths_resolve_and_missing_overrides_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'ola_weights').mkdir()
            (root/'ola_weights/large-v3.pt').touch()
            with patch.dict(os.environ, {'PSYMM_WEIGHTS':tmp},clear=True):
                config=SimpleNamespace(speech_encoder='/previous-host/checkpoints/large-v3.pt')
                module.resolve_encoder_paths(config)
                self.assertEqual(config.speech_encoder,str(root/'ola_weights/large-v3.pt'))
                with self.assertRaises(FileNotFoundError):
                    module.resolve_encoder_paths(SimpleNamespace(music_encoder='/previous-host/checkpoints/missing.pt'))
            with patch.dict(os.environ, {'PSYMM_OLA_SPEECH_ENCODER':str(root/'missing.pt')},clear=True):
                with self.assertRaises(FileNotFoundError):
                    module.resolve_encoder_paths(SimpleNamespace(speech_encoder='large-v3'))
