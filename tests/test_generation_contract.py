import copy
import json
import tempfile
import unittest
from pathlib import Path

from psymm_generation.core import clean_text, emotion, normalized_dialogue
from psymm_generation.references import catalog_from_metadata, identity_pair
from psymm_generation.cli import reusable, database, verified_record


class GenerationContract(unittest.TestCase):
    def test_roles_preserve_content_and_emotion_provenance(self):
        raw = dict(conversation_id=3, emotion_type='anxiety', dialog=[
            dict(speaker='seeker', content='I take 10mg—at 8:30. 我很焦虑。'),
            dict(speaker='supporter', content='Tell me more.'),
            dict(speaker='seeker', content='A little better.', emotion='surprise')])
        saved = copy.deepcopy(raw)
        turns = list(normalized_dialogue('extes', raw, 0))[0]['turns']
        self.assertEqual(raw, saved)
        self.assertEqual(turns[0]['text'], raw['dialog'][0]['content'])
        self.assertEqual([t['emotion'] for t in turns], ['fear', 'happy', 'surprised'])
        self.assertEqual(turns[1]['emotion_origin'], 'original_source_supporter_policy')

    def test_fixed_distinct_identities_independent_of_turn_order(self):
        catalog = {'16': [], '42': [], '18': []}
        first = identity_pair(catalog, 'extes_000001', 17)
        self.assertEqual(first, identity_pair(catalog, 'extes_000001', 17))
        self.assertNotEqual(first['seeker'], first['supporter'])
        self.assertEqual(first, identity_pair(dict(reversed(list(catalog.items()))), 'extes_000001', 17))

    def test_reference_pair_has_same_segment_and_profile(self):
        records = [dict(conversation_id='00001', speaker_profile={'ID':16},
                        listener_profile={'ID':42}, turns=[dict(dialogue_history=[
                            dict(index=0, role='speaker', utterance='hello'),
                            dict(index=1, role='listener', utterance='hi')])])]
        catalog = catalog_from_metadata(records, Path('/data/AvaMERG'))
        r = catalog['16'][0]
        self.assertEqual(Path(r['audio']).stem, Path(r['video']).stem)
        self.assertEqual(r['segment'], 'dia00001utt0_16')

    def test_unknown_emotion_rejected_and_disgust_kept_separate(self):
        self.assertEqual(emotion('disgust'), 'disgusted')
        self.assertEqual(emotion('surprise'), 'surprised')
        with self.assertRaises(ValueError):
            emotion('unreviewed_label')
        self.assertEqual(clean_text('剂量10mg，早上8点\n不要忘记。'), '剂量10mg，早上8点 不要忘记。')

    def test_psyqa_policy_preserves_answer_association(self):
        row = dict(questionID=8, emotion_type='sad', question='问题', description='描述',
                   answers=[{'answer_text':'回答一'}, {'answer_text':'回答二'}])
        units = list(normalized_dialogue('psyqa', row, 0, 'all'))
        self.assertEqual([d['answer_index'] for d in units], [0,1])
        self.assertNotEqual(units[0]['dialogue_id'], units[1]['dialogue_id'])
        self.assertEqual(units[1]['turns'][1]['text'], '回答二')
        self.assertEqual(len(list(normalized_dialogue('psyqa', row, 0))), 1)

    def test_private_reasoning_and_system_prompt_never_spoken(self):
        openr1 = dict(post_id=1, emotion_type='happy', conversation=[
            dict(patient='Hello', counselor_content='Welcome', counselor_think='PRIVATE')])
        turns = list(normalized_dialogue('openr1', openr1, 0))[0]['turns']
        self.assertEqual([t['text'] for t in turns], ['Hello','Welcome'])
        psydial = dict(messages=[dict(role='system', content='PRIVATE'),
                                dict(role='user', content='你好')])
        turns = list(normalized_dialogue('psydial', psydial, 0))[0]['turns']
        self.assertEqual(len(turns),1)
        self.assertEqual(turns[0]['turn_index'],1)

    def test_resume_rejects_failed_stale_and_corrupt_files(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'bad.wav'
            path.write_bytes(b'not audio')
            self.assertFalse(reusable(('old','complete','hash'), 'new', path, 'audio'))
            self.assertFalse(reusable(('new','failed','hash'), 'new', path, 'audio'))
            from psymm_generation.core import file_hash
            self.assertFalse(reusable(('new','complete',file_hash(path)), 'new', path, 'audio'))

    def test_validation_rejects_changed_code_input_and_audio_dependency(self):
        from psymm_generation.core import atomic_json, digest, file_hash
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            media = run/'sample.wav'
            media.write_bytes(b'fixture media')
            image = run/'portrait.png'
            image.write_bytes(b'fixture image')
            item = dict(asset_id='a', text='source text', emotion='neutral', reference=dict(
                audio=str(media), audio_sha256=file_hash(media),
                portrait=str(image), portrait_sha256=file_hash(image)))
            info = dict(code={'worker':'v1'},weights={'model':'v1'})
            config_hash = digest(dict(info, options={},stage='audio'))
            atomic_json(run/'configurations'/(config_hash+'.json'),
                        dict(inventory=info, options={},stage='audio'))
            db = database(run)
            db.execute('INSERT INTO media VALUES (?,?,?,?,?,?,?,?)',
                ('a','audio',digest([item,config_hash,None]),'complete',str(media),
                 file_hash(media),json.dumps(dict(configuration_hash=config_hash)),0))
            db.commit()
            verified_record(db,run,item,'audio',info)


            video_config_hash = digest(dict(info,options={},stage='video'))
            self.assertNotEqual(config_hash,video_config_hash)
            atomic_json(run/'configurations'/(video_config_hash+'.json'),
                        dict(inventory=info,options={},stage='video'))
            db.execute('INSERT INTO media VALUES (?,?,?,?,?,?,?,?)',
                ('a','video',digest([item,video_config_hash,file_hash(media)]),'complete',str(media),
                 file_hash(media),json.dumps(dict(configuration_hash=video_config_hash)),0))
            db.commit()
            verified_record(db,run,item,'audio',info)
            verified_record(db,run,item,'video',info)
            with self.assertRaises(ValueError):
                verified_record(db,run,item,'audio',dict(info,code={'worker':'v2'}))
            with self.assertRaises(ValueError):
                verified_record(db,run,dict(item,text='changed'),'audio',info)
            db.close()


if __name__ == '__main__':
    unittest.main()
