import unittest
from experiments.audit_e0 import group_keys, grouped_assignments
from psymm_generation.core import normalized_dialogue
from build_ola_dataset import response_rows


def keys(source, record, index=0):
    return group_keys(source, record, list(normalized_dialogue(source,record,index)))


class E0Isolation(unittest.TestCase):
    def test_question_variants_keep_different_answers_together(self):
        first=dict(questionID=1,question='How can I cope?',description='',
                   answers=[dict(answer_text='One answer')])
        second=dict(questionID=2,question='How can I cope?',description='',
                    answers=[dict(answer_text='Different answer')])
        groups=list(grouped_assignments([dict(keys=keys('psyqa',x)) for x in (first,second)],7))
        self.assertEqual(len(groups),1)

    def test_exact_dialogue_duplicates_cross_source_and_order(self):
        record=dict(dialog=[dict(speaker='seeker',content='I feel sad.'),
                            dict(speaker='supporter',content='I am listening.')])
        a=keys('esconv',record)
        changed=dict(dialog=[dict(speaker='seeker',content='  I FEEL sad. '),
                             dict(speaker='supporter',content='I am listening.')])
        b=keys('extes',changed)
        left=list(grouped_assignments([dict(keys=a),dict(keys=b)],11))
        right=list(grouped_assignments([dict(keys=b),dict(keys=a)],11))
        self.assertEqual(len(left),1)
        self.assertEqual(left[0][2:],right[0][2:])

    def test_transitive_shared_keys_cannot_cross_splits(self):
        nodes=[dict(keys=['id:a','dialogue:x']),dict(keys=['id:b','dialogue:x']),
               dict(keys=['id:b','dialogue:y']),dict(keys=['id:c'])]
        result=list(grouped_assignments(nodes,42))
        self.assertEqual(sorted(len(g[0]) for g in result),[1,3])

    def test_short_greeting_and_null_ids_do_not_merge_distinct_dialogues(self):
        def record(answer):
            return dict(conversation_id=None,dialog=[dict(speaker='seeker',content='Hello'),
                dict(speaker='supporter',content=answer)])
        a,b=[keys('esconv',record(answer)) for answer in ('Answer A','Answer B')]
        self.assertFalse(set(a)&set(b))

    def test_same_original_id_keeps_persona_or_emotion_variants_together(self):
        def record(text):
            return dict(conversation_id='original-5',dialog=[dict(speaker='seeker',content=text)])
        result=list(grouped_assignments([dict(keys=keys('esconv',record(t)))
            for t in ('Variant A','Variant B')],42))
        self.assertEqual(len(result),1)

    def test_no_target_or_future_turn_in_conditioning(self):
        items=[]
        turns=[('seeker','QUESTION_1'),('supporter','TARGET_1'),
               ('seeker','FUTURE_QUESTION'),('supporter','FUTURE_TARGET')]
        for i,(role,text) in enumerate(turns):
            items.append(dict(dialogue_id='d',asset_id=f'a{i}',turn_index=i,role=role,
                text=text,source='fixture',source_id=1,emotion='sad',
                media={s:dict(path=f'a{i}.{ext}') for s,ext in [('audio','wav'),('video','mp4')]}))
        rows,skipped=response_rows(items)
        self.assertFalse(skipped)
        context=rows[0]['conversations'][0]['value']
        for secret in ('TARGET_1','FUTURE_QUESTION','FUTURE_TARGET'):
            self.assertNotIn(secret,context)
        self.assertEqual(rows[0]['video'],'a0.mp4')
        self.assertEqual(rows[1]['video'],'a2.mp4')
        self.assertIn('TARGET_1',rows[1]['conversations'][0]['value'])


if __name__=='__main__':
    unittest.main()
