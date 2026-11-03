import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from psymm_generation.core import normalized_dialogue, file_hash
from psymm_generation.references import (emotion_pool_from_filenames, identity_pair,
                                         select_dialogue_references, select_anchor,
                                         verify_reference)
from psymm_generation.cli import read_plan


class EmotionPoolContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pool_dir = self.root / 'emo'
        self.pool_dir.mkdir()
        self.catalog = {}
        for person in ('10', '20', '30'):
            refs = []
            for index in range(3):
                stem = f'dia00001utt{index}_{person}'
                audio = self.root / (stem + '.wav')
                audio.write_bytes(b'reference audio ' + stem.encode())
                refs.append(dict(identity=person, segment=stem, audio=str(audio),
                                 video=str(self.root / (stem + '.mp4'))))
            self.catalog[person] = refs

        for label, people, index in [('fear', ('10', '20'), 0),
                                     ('happy', ('20', '30'), 1),
                                     ('surprise', ('10', '30'), 2)]:
            folder = self.pool_dir / label
            folder.mkdir()
            for person in people:
                ref = self.catalog[person][index]
                (folder / (ref['segment'] + '.wav')).write_bytes(Path(ref['audio']).read_bytes())
        self.pool = emotion_pool_from_filenames(self.catalog, self.pool_dir)

    @staticmethod
    def dialogue(name='extes_1', seeker='fear', supporter='happy'):
        return dict(dialogue_id=name, turns=[dict(role=role, emotion=label) for role, label in
                    [('seeker', seeker), ('supporter', supporter), ('seeker', seeker)]])

    @staticmethod
    def fake_frame(command, timeout=None):
        Path(command[-1]).write_bytes(b'portrait fixture' * 100)

    def anchors(self, dialogue):
        with patch('psymm_generation.references.probe_audio',
                   return_value=dict(seconds=3., silence_fraction=.1)), \
             patch('psymm_generation.references.checked', side_effect=self.fake_frame):
            return select_dialogue_references(self.catalog, self.pool, dialogue,
                                               self.root / 'references', 17)

    def test_emotion_selection_fixes_two_paired_anchors_for_whole_dialogue(self):
        d = self.dialogue()
        refs = self.anchors(d)
        self.assertNotEqual(refs['seeker']['identity'], refs['supporter']['identity'])
        self.assertEqual(refs, self.anchors(d))
        for turn in d['turns']:
            ref = refs[turn['role']]
            self.assertEqual(ref['reference_emotion'], turn['emotion'])
            self.assertEqual(Path(ref['audio']).parent.name, turn['emotion'])
            self.assertEqual(Path(ref['audio']).stem, Path(ref['video']).stem)
            self.assertEqual(Path(ref['portrait']).stem, ref['segment'])
            verify_reference(dict(turn, reference=ref))

    def test_neutral_keeps_identity_conversion_without_sampling_neutral_pool(self):

        (self.pool_dir / 'neutral').mkdir()
        pool = emotion_pool_from_filenames(self.catalog, self.pool_dir)
        self.assertNotIn('neutral', pool)
        refs = self.anchors(self.dialogue(seeker='neutral'))
        ref = refs['seeker']
        self.assertIsNone(ref['reference_emotion'])
        self.assertEqual(ref['reference_source'], 'avamerg_identity_anchor')
        verify_reference(dict(emotion='neutral', reference=ref))
        with self.assertRaises(ValueError):
            verify_reference(dict(emotion='neutral', reference=refs['supporter']))

    def test_surprise_alias_preserves_actual_pool_path(self):
        ref = self.anchors(self.dialogue(seeker='surprised'))['seeker']
        self.assertEqual(ref['reference_emotion'], 'surprised')
        self.assertEqual(Path(ref['audio']).parent.name, 'surprise')

    def test_ambiguous_pool_memberships_are_excluded_and_recorded(self):
        stem = self.catalog['10'][0]['segment']
        (self.pool_dir / 'happy' / (stem + '.wav')).write_bytes(b'ambiguous copy')
        audit = {}
        pool = emotion_pool_from_filenames(self.catalog, self.pool_dir, audit=audit)
        self.assertEqual(audit['excluded_ambiguous_references'][stem], ['fear', 'happy'])
        self.assertNotIn(stem, [r['segment'] for people in pool.values()
                               for refs in people.values() for r in refs])

    def test_missing_emotion_and_identity_conflicts_never_fall_back(self):
        with self.assertRaisesRegex(ValueError, 'no neutral fallback'):
            self.anchors(self.dialogue(seeker='contempt'))
        with self.assertRaisesRegex(ValueError, 'two distinct'):
            identity_pair(self.catalog, 'x', 17, dict(seeker=['10'], supporter=['10']))
        d = self.dialogue()
        d['turns'][-1]['emotion'] = 'sad'
        with self.assertRaisesRegex(ValueError, 'fixed audio anchor'):
            self.anchors(d)

    def test_reference_cache_is_scoped_to_dialogue_and_role(self):
        with patch('psymm_generation.references.probe_audio',
                   return_value=dict(seconds=3., silence_fraction=.1)), \
             patch('psymm_generation.references.checked', side_effect=self.fake_frame):
            for scope in ('dialogue_1:seeker', 'dialogue_2:seeker'):
                ref = select_anchor(self.catalog, '10', self.root / 'cache', 17, scope=scope)
                self.assertEqual(ref['dialogue_scope'], scope)
        self.assertEqual(len(list((self.root / 'cache').glob('anchor_*.json'))), 2)

    def test_unusable_person_is_replaced_inside_same_emotion_pool_before_plan(self):
        d = self.dialogue()
        first = identity_pair(self.catalog, d['dialogue_id'], 17,
                              dict(seeker=sorted(self.pool['fear']),
                                   supporter=sorted(self.pool['happy'])))['seeker']
        def choose(catalog, person, cache, seed, scope, target_emotion, validate_speech=False):
            if scope.endswith(':seeker') and person == first:
                raise ValueError('No usable paired reference')
            return dict(identity=person, reference_emotion=target_emotion)
        with patch('psymm_generation.references.select_anchor', side_effect=choose):
            refs = select_dialogue_references(self.catalog, self.pool, d, self.root, 17)
        self.assertNotEqual(refs['seeker']['identity'], first)
        self.assertIn(refs['seeker']['identity'], self.pool['fear'])
        self.assertIn(refs['supporter']['identity'], self.pool['happy'])
        self.assertNotEqual(refs['seeker']['identity'], refs['supporter']['identity'])

    def test_reference_label_or_file_changes_are_rejected(self):
        ref = self.anchors(self.dialogue())['seeker']
        with self.assertRaises(ValueError):
            verify_reference(dict(emotion='happy', reference=ref))
        Path(ref['audio']).write_bytes(b'changed reference')
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            verify_reference(dict(emotion='fear', reference=ref))

    def test_source_specific_supporter_policy_and_explicit_annotations(self):
        for source, expected in [('extes', 'happy'), ('psydial', 'happy'), ('esconv', 'neutral')]:
            row = dict(dialog=[dict(speaker='supporter', content='Hello')])
            turn = list(normalized_dialogue(source, row, 0))[0]['turns'][0]
            self.assertEqual(turn['emotion'], expected)
        row = dict(input='Hello', output='Welcome')
        self.assertEqual(list(normalized_dialogue('psych8k', row, 0))[0]['turns'][1]['emotion'], 'sad')
        row = dict(dialog=[dict(speaker='supporter', content='Hello', emotion='neutral')])
        turn = list(normalized_dialogue('extes', row, 0))[0]['turns'][0]
        self.assertEqual((turn['emotion'], turn['emotion_origin']), ('neutral', 'turn_annotation'))

    def test_old_plan_cannot_silently_reuse_audio_without_emotion_pool(self):
        (self.root / 'plan.meta.json').write_text(json.dumps(dict(schema_version=1)))
        with self.assertRaisesRegex(ValueError, 'prepare a new'):
            read_plan(self.root)


if __name__ == '__main__':
    unittest.main()
