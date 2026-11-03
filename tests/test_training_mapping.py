import unittest
from build_ola_dataset import response_rows


class TrainingMapping(unittest.TestCase):
    def test_conditioning_media_belongs_to_seeker_not_target(self):
        items=[]
        for i,(role,text) in enumerate([('seeker','question'),('supporter','SECRET ANSWER')]):
            items.append(dict(dialogue_id='d',asset_id=f'a{i}',turn_index=i,role=role,
                text=text,source='fixture',source_id=1,emotion='sad',
                media={s:dict(path=f'{role}.{ext}') for s,ext in [('audio','wav'),('video','mp4')]}))
        rows,skips=response_rows(items)
        self.assertFalse(skips)
        self.assertEqual(rows[0]['audio'],'seeker.wav')
        self.assertEqual(rows[0]['video'],'seeker.mp4')
        self.assertNotIn('SECRET ANSWER',rows[0]['conversations'][0]['value'])
        self.assertTrue(rows[0]['conversations'][0]['value'].startswith('<speech><video>'))
        self.assertEqual(rows[0]['conversations'][1]['value'],'SECRET ANSWER')
