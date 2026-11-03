"""CPU checks for legacy WAV -> video compatibility, without model or CUDA imports."""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import legacy_video as lv


class LegacyVideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / 'infer_outputs_extes'
        self.audio = self.dataset / 'OpenVoiceEMO_outputs'
        self.images = self.root / 'infer_outputs_esconv/avamerg_test_images'
        self.priors = self.root / 'DICE-Talk-main/examples/emo'
        for directory in (self.audio, self.images, self.priors):
            directory.mkdir(parents=True)
        (self.priors / 'surprised.npy').write_bytes(b'prior')
        (self.priors / 'fear.npy').write_bytes(b'prior')
        from PIL import Image
        Image.new('RGB', (128, 128), 'green').save(self.images / 'dia0utt0_7.jpg')
        Image.new('RGB', (128, 128), 'green').save(self.images / 'dia0utt1_8.jpg')
        self.mapping = {
            'ExTES_00000_T000.wav': dict(post_id=0, role='seeker', reference_audio='dia0utt0_7.wav', emotion='fear'),
            'ExTES_00000_T001.wav': dict(post_id=0, role='supporter', reference_audio='dia0utt1_8.wav', emotion='surprise')}
        self.mapfile = self.dataset / 'audio_reference_map_emo.json'
        self.mapfile.write_text(json.dumps(self.mapping))
        import numpy as np
        import soundfile as sf
        wav = .2 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)
        for name in self.mapping:
            sf.write(str(self.audio / name), wav, 16000)
        (self.audio / 'tmp_worker0.wav').write_bytes(b'temporary')
        self.root_patch = patch.object(lv, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.args = types.SimpleNamespace(dataset_dir=self.dataset, audio_map=None,
                                         audio_dir=None, image_dir=None, limit=0)
        self.environment = patch.dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', ''))
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_readonly_plan_exact_reference_and_identity_diagnostic(self):
        with patch.object(lv, 'file_hash', side_effect=AssertionError('must not open media during planning')):
            items, summary = lv.plan(self.args)
        self.assertEqual(summary['requested'], 2)
        self.assertEqual(summary['unmapped_wavs'], 1)
        self.assertEqual(summary['dialogues_with_identity_changes'], 0)
        lv.require_fixed_identities(summary)
        self.assertEqual(items[1]['emotion'], 'surprised')
        self.assertEqual(Path(items[1]['portrait']).name, 'dia0utt1_8.jpg')
        self.assertFalse((self.dataset / 'DICE_Talk_outputs').exists())

    def test_drift_outside_limit_and_missing_audio_cannot_bypass_guard(self):
        self.mapping['ExTES_00000_T002.wav'] = dict(post_id=0, role='seeker',
            reference_audio='dia0utt1_8.wav', emotion='fear')
        self.mapfile.write_text(json.dumps(self.mapping))
        self.args.limit = 1
        items, summary = lv.plan(self.args)
        self.assertEqual(len(items), 1)
        self.assertEqual(summary['missing_audio'], 1)
        self.assertEqual(summary['dialogues_with_identity_changes'], 1)
        self.assertEqual(summary['dialogues_with_shared_role_identity'], 1)
        with self.assertRaisesRegex(ValueError, 'regenerate audio and video'):
            lv.require_fixed_identities(summary)
        self.assertFalse((self.dataset / 'DICE_Talk_outputs').exists())

    def test_invalid_legacy_map_stops_before_output_writes_or_workers(self):
        self.mapping['ExTES_00000_T001.wav']['role'] = 'seeker'
        self.mapfile.write_text(json.dumps(self.mapping))
        with patch.object(sys, 'argv', ['infer_video.py', '--gpu', '3']), \
             patch.object(lv.mp, 'get_context') as processes, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'fixed two-role'):
                lv.main(self.dataset)
            processes.assert_not_called()
        self.assertFalse((self.dataset / 'DICE_Talk_outputs').exists())

    def test_unknown_role_missing_dialogue_and_inconsistent_reference_path_rejected(self):
        for patch_entry in (dict(role='unknown'), dict(post_id=None),
                            dict(reference_path='/pool/dia9utt0_99.wav')):
            mapping = json.loads(json.dumps(self.mapping))
            mapping['ExTES_00000_T000.wav'].update(patch_entry)
            report = lv.identity_audit(mapping)
            self.assertEqual(report['invalid_identity_records'], 1)
            with self.assertRaises(ValueError):
                lv.require_fixed_identities(report)

    def test_missing_assets_counted_without_reselecting_reference(self):
        (self.audio / 'ExTES_00000_T001.wav').unlink()
        (self.images / 'dia0utt0_7.jpg').unlink()
        items, summary = lv.plan(self.args)
        self.assertEqual(items, [])
        self.assertEqual(summary['missing_audio'], 1)
        self.assertEqual(summary['missing_portrait'], 1)
        self.assertEqual(json.loads(self.mapfile.read_text()), self.mapping)

    def test_unknown_emotion_fails_without_neutral_fallback(self):
        self.mapping['ExTES_00000_T000.wav']['emotion'] = 'unmapped'
        self.mapfile.write_text(json.dumps(self.mapping))
        with self.assertRaisesRegex(ValueError, 'Unknown emotion'):
            lv.plan(self.args)

    def test_generate_resume_corruption_repair_and_failure_retry(self):
        items, _ = lv.plan(self.args)
        cache = self.dataset / '.DICE_Talk_outputs_generation'
        cache.mkdir()
        out = self.dataset / 'DICE_Talk_outputs'
        out.mkdir()
        settings = dict(gpu=3, cache=str(cache), output=str(out), seed=42,
                        options=dict(steps=25, ref_scale=3., emo_scale=6.,
                                     video_segment_seconds=20, _configuration_hash='cpu-test'))
        calls, fail = [], [True]
        class FakeDICE:
            def __init__(self, **kw):
                pass
            def preprocess(self, image, **kw):
                return dict(face_num=1, crop_bbox=[0, 0, 128, 128])
            def crop_image(self, source, target, bbox):
                shutil.copyfile(source, target)
            def process(self, portrait, audio, prior, output, **kw):
                calls.append(audio)
                if fail[0] and audio.endswith('T001.wav'):
                    raise RuntimeError('reproducible item failure')
                subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                                'color=c=green:s=128x128:r=25:d=1', '-i', audio,
                                '-c:v', 'libx264', '-threads', '1', '-c:a', 'aac', output], check=True)
                return 0
        fake = types.SimpleNamespace(DICE_Talk=FakeDICE)
        with patch.dict(sys.modules, dice_talk=fake), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(lv.run_items(0, items, settings), 1)
            self.assertEqual(dict(lv.database_counts(cache / 'state.sqlite')), {'complete': 1, 'failed': 1})
            fail[0] = False
            self.assertEqual(lv.run_items(0, items, settings), 0)
            self.assertEqual(len(calls), 3)
            self.assertEqual(lv.run_items(0, items, settings), 0)
            self.assertEqual(len(calls), 3)
            (out / 'ExTES_00000_T000.mp4').write_bytes(b'corrupt')
            self.assertEqual(lv.run_items(0, items, settings), 0)
            self.assertEqual(len(calls), 4)
        records = [json.loads(x) for x in (cache / 'reference_map_00.jsonl').read_text().splitlines()]
        self.assertEqual(records[0]['original_reference'], self.mapping[items[0]['name']])
        self.assertEqual(records[1]['identity'], '8')

        self.assertEqual(json.loads(self.mapfile.read_text()), self.mapping)


if __name__ == '__main__':
    unittest.main()
