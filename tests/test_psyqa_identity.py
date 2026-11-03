import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from psyqa_identity import validate_items, check_maps, check_audio_receipts, preflight, check_run


def sample_items(dialogue='psyqa_000001_abc_a00', answer_index=0):
    items = []
    for turn, (role, person, emotion) in enumerate((('seeker', '10', 'fear'),
                                                   ('supporter', '20', 'happy'))):
        stem = f'dia00001utt{turn}_{person}'
        ref = dict(identity=person, segment=stem, audio=f'/pool/{emotion}/{stem}.wav',
                   video=f'/video/{stem}.mp4', portrait=f'/run/references/{stem}.png',
                   audio_sha256='audio-'+person, portrait_sha256='portrait-'+person,
                   reference_emotion=emotion, dialogue_scope=f'{dialogue}:{role}')
        items.append(dict(source='psyqa', source_index=1, answer_index=answer_index,
                          asset_id=f'{dialogue}_T{turn:04d}', dialogue_id=dialogue,
                          role=role, emotion=emotion, reference=ref))
    return items


class PsyQAIdentityContract(unittest.TestCase):
    def test_extes_empty_turns_only_are_removed_and_original_indices_recorded(self):
        import generate_multimodal as entry
        import sys
        original = [dict(conversation_id=10, emotion_type='fear', dialog=[
            dict(speaker='seeker', content='I take 10mg at 8:30.'),
            dict(speaker='supporter', content=' \n\t'),
            dict(speaker='supporter', content='Please continue.'),
            dict(speaker='seeker', content='\u200b'),
            dict(speaker='seeker', content='Still worried.')])]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, run = root/'original.json', root/'run'
            source.write_text(json.dumps(original))
            before = source.read_bytes()
            with patch.object(sys, 'argv', ['generate_multimodal.py', 'prepare', '--source',
                    'extes', '--input='+str(source), '--run', str(run)]):
                entry.prepare_extes_nonempty_input()
                self.assertEqual(sys.argv[sys.argv.index('--input')+1], str(run/'source_nonempty.json'))
            derived = json.loads((run/'source_nonempty.json').read_text())
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(derived[0]['dialog'], [original[0]['dialog'][i] for i in (0, 2, 4)])
            report = json.loads((run/'empty_turns_skipped.json').read_text())
            self.assertEqual((report['skipped_turn_count'], report['affected_dialogues']), (2, 1))
            self.assertEqual([r['original_turn_index'] for r in report['skipped']], [1, 3])
            self.assertEqual(report['turn_index_maps'][0]['retained_original_turn_indices'], [0, 2, 4])
            (run/'plan.meta.json').write_text('{}')
            with patch.object(sys, 'argv', ['entry', 'prepare', '--source', 'extes',
                    '--input', str(source), '--run', str(run)]):
                with self.assertRaisesRegex(ValueError, 'already prepared'):
                    entry.prepare_extes_nonempty_input()

    def test_empty_turn_policy_does_not_change_other_sources_or_mask_invalid_text(self):
        import generate_multimodal as entry
        import sys
        from psymm_generation.core import normalized_dialogue
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for source in ('psyqa', 'psydial'):
                with patch.object(sys, 'argv', ['entry', 'prepare', '--source', source,
                        '--run', str(root/source)]):
                    entry.prepare_extes_nonempty_input()
                self.assertFalse((root/source).exists())
            path = root/'bad.json'
            path.write_text(json.dumps([dict(dialog=[dict(speaker='seeker', content=None)])]))
            run = root/'extes'
            with patch.object(sys, 'argv', ['entry', 'prepare', '--source', 'extes',
                    '--input', str(path), '--run', str(run)]):
                with self.assertRaisesRegex(ValueError, 'Text must be a string'):
                    entry.prepare_extes_nonempty_input()
            self.assertFalse(run.exists())

    def test_extes_missing_role_quarantine_keeps_source_indices_and_original_record(self):
        import generate_multimodal as entry
        import sys
        from psymm_generation.core import normalized_dialogue
        original = [dict(post_id=1, dialog=[dict(speaker='seeker', content='Hello'),
                                          dict(speaker='supporter', content=' ')]),
                    dict(post_id=2, dialog=[dict(speaker='seeker', content='Hello again'),
                                          dict(speaker='supporter', content='Welcome')])]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, run = root/'source.json', root/'run'
            path.write_text(json.dumps(original))
            with patch.object(sys, 'argv', ['entry', 'prepare', '--source', 'extes',
                    '--input', str(path), '--run', str(run)]):
                entry.prepare_extes_nonempty_input()
                self.assertEqual(sys.argv[-2:], ['--invalid-record-policy', 'quarantine'])
            derived = json.loads((run/'source_nonempty.json').read_text())
            self.assertEqual(len(derived), 2)
            with self.assertRaisesRegex(ValueError, 'No dialogue turns'):
                list(normalized_dialogue('extes', derived[0], 0))
            dialogue = next(normalized_dialogue('extes', derived[1], 1))
            self.assertEqual(dialogue['source_index'], 1)
            self.assertEqual(derived[1], original[1])
            quarantine = json.loads((run/'quarantined_source_records.json').read_text())
            self.assertEqual(quarantine['records'][0]['original_record'], original[0])
            self.assertEqual(quarantine['records'][0]['missing_roles'], ['supporter'])
            self.assertEqual(json.loads(path.read_text()), original)

    def test_multiturn_psydial_and_extes_keep_role_binding_for_entire_dialogue(self):
        from psymm_generation.core import normalized_dialogue
        for source in ('psydial', 'extes'):
            if source == 'psydial':
                record = dict(emotion_type='fear', messages=[
                    dict(role='system', content='Do not speak this'),
                    dict(role='user', content='我今天很焦虑，8点吃了10mg。'),
                    dict(role='assistant', content='你好，请继续。'),
                    dict(role='user', content='最近一直如此。'),
                    dict(role='assistant', content='我们一起想办法。')])
            else:
                record = dict(emotion_type='fear', dialog=[
                    dict(speaker=r, content='Hello') for r in
                    ('seeker', 'supporter', 'seeker', 'supporter')])
            dialogue = next(normalized_dialogue(source, record, 1))
            base = sample_items(dialogue['dialogue_id'])
            refs = {x['role']: x['reference'] for x in base}
            items = [dict(source=source, source_index=1, answer_index=None,
                          dialogue_id=dialogue['dialogue_id'],
                          asset_id=f"{dialogue['dialogue_id']}_T{t['turn_index']:04d}",
                          **t, reference=copy.deepcopy(refs[t['role']])) for t in dialogue['turns']]
            report = validate_items(items)
            self.assertEqual((report['scope'], report['turns']), ('whole_dialogue', 4))
            if source == 'psydial':
                self.assertEqual(items[0]['turn_index'], 1)
                self.assertIn('10mg', items[0]['text'])
            items[2]['reference']['audio_sha256'] = 'another-reference'
            with self.assertRaisesRegex(ValueError, 'changes within dialogue role'):
                validate_items(items)

    def test_dataset_entry_uses_guard_and_rejects_wrong_dataset(self):
        import generate_multimodal as entry
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            items = sample_items()
            (run/'plan.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in items))
            with patch.object(sys, 'argv', ['infer_video.py', '--run', str(run)]), \
                 patch.object(entry, 'cli_main') as model:
                with self.assertRaisesRegex(ValueError, 'cannot use'):
                    entry.dataset_main('video', 'psydial')
                model.assert_not_called()
            items[0]['reference']['portrait'] = items[1]['reference']['portrait']
            with patch.object(sys, 'argv', ['infer_video.py', '--run', str(run)]), \
                 patch('psymm_generation.cli.read_plan', return_value=items), \
                 patch.object(entry, 'cli_main') as model:
                with self.assertRaisesRegex(ValueError, 'Unpaired reference'):
                    entry.dataset_main('video', 'psyqa')
                model.assert_not_called()

    def test_paired_distinct_roles_and_independent_answer_units(self):
        items = sample_items()
        second = sample_items('psyqa_000001_abc_a01', 1)

        report = validate_items(items+second)
        self.assertEqual((report['dialogue_count'], report['turns']), (2, 4))
        self.assertEqual(report['scope'], 'independent_question_answer')
        with self.assertRaisesRegex(ValueError, 'empty'):
            validate_items([])

    def test_swapped_portrait_or_reference_audio_rejected(self):
        for field in ('portrait', 'audio', 'video'):
            items = sample_items()
            items[0]['reference'][field] = items[1]['reference'][field]
            with self.assertRaisesRegex(ValueError, 'Unpaired reference'):
                validate_items(items)

    def test_role_cannot_change_identity_or_anchor_within_dialogue(self):
        items = sample_items()
        extra = copy.deepcopy(items[0])
        extra['asset_id'] += '_repeat'
        extra['reference']['portrait_sha256'] = 'different_portrait'
        with self.assertRaisesRegex(ValueError, 'changes within dialogue role'):
            validate_items(items+[extra])

    def test_roles_must_be_distinct_and_bound_to_correct_scope(self):
        items = sample_items()
        items[1]['reference'] = copy.deepcopy(items[0]['reference'])
        items[1]['reference'].update(dialogue_scope=items[1]['dialogue_id']+':supporter',
                                     reference_emotion='happy')
        with self.assertRaisesRegex(ValueError, 'two distinct'):
            validate_items(items)
        items = sample_items()
        items[0]['reference']['dialogue_scope'] = 'another-dialogue:seeker'
        with self.assertRaisesRegex(ValueError, 'another dialogue/role'):
            validate_items(items)

    def test_metadata_identity_cannot_disagree_with_segment_profile(self):
        items = sample_items()
        items[0]['reference']['identity'] = '99'
        with self.assertRaisesRegex(ValueError, 'segment/profile identity mismatch'):
            validate_items(items)

    def test_neutral_uses_no_emotion_pool_label(self):
        items = sample_items()
        items[0]['emotion'] = 'neutral'
        items[0]['reference']['reference_emotion'] = None
        validate_items(items)
        items[0]['reference']['reference_emotion'] = 'fear'
        with self.assertRaisesRegex(ValueError, 'emotion mismatch'):
            validate_items(items)

    def test_old_maps_and_new_maps_with_other_identity_rejected(self):
        items = sample_items()
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            mapping = run/'ref_map.json'
            mapping.write_text(json.dumps({'1': {'speaker_ref': 'wrong.wav'}}))
            with self.assertRaisesRegex(ValueError, 'Legacy identity map'):
                check_maps(run, items)
            records = {}
            for item in items:
                ref = item['reference']
                records[item['asset_id']] = dict(identity=ref['identity'], segment=ref['segment'],
                    role=item['role'], emotion=item['emotion'], reference_image=ref['portrait'],
                    sha256=ref['portrait_sha256'])
            mapping.write_text(json.dumps(dict(schema_version=1, records=records)))
            check_maps(run, items)
            records[items[0]['asset_id']]['reference_image'] = items[1]['reference']['portrait']
            mapping.write_text(json.dumps(dict(schema_version=1, records=records)))
            with self.assertRaisesRegex(ValueError, 'disagrees with plan'):
                check_maps(run, items)

    def test_completed_audio_must_use_portraits_paired_voice_reference(self):
        items = sample_items()
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            with sqlite3.connect(str(run/'state.sqlite')) as db:
                db.execute('CREATE TABLE media (asset TEXT,stage TEXT,status TEXT,details TEXT)')
                ref = items[0]['reference']
                receipt = dict(reference_audio=ref['audio'], reference_audio_sha256=ref['audio_sha256'])
                db.execute('INSERT INTO media VALUES (?,?,?,?)',
                           (items[0]['asset_id'], 'audio', 'complete', json.dumps(receipt)))
            self.assertEqual(check_audio_receipts(run, items), 1)
            with sqlite3.connect(str(run/'state.sqlite')) as db:
                receipt['reference_audio'] = items[1]['reference']['audio']
                db.execute('UPDATE media SET details=?', (json.dumps(receipt),))
            with self.assertRaisesRegex(ValueError, 'another identity/reference'):
                check_audio_receipts(run, items)

    def test_video_entry_checks_before_model_execution_and_exports_bindings(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            items = sample_items()
            (run/'plan.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in items))
            with patch('psymm_generation.cli.read_plan', return_value=items):
                preflight(['video', '--run', str(run), '--gpu', '3', '--keep-going'])
                manifest = json.loads((run/'dialogue_identity_map.json').read_text())
                self.assertEqual(manifest['dialogue_count'], 1)
                items[0]['reference']['portrait'] = items[1]['reference']['portrait']
                with self.assertRaisesRegex(ValueError, 'Unpaired reference'):
                    preflight(['video', '--run', str(run), '--gpu', '3'])
                self.assertEqual(json.loads((run/'identity_validation.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
