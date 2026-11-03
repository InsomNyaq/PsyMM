"""Small independent fixtures; never write synthetic labels into real E1 bundles."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

SCRIPT = Path(__file__).with_name('e1_emotion_validation.py')
if not SCRIPT.exists():
    SCRIPT = Path(__file__).resolve().parents[1]/'experiments/e1_emotion_validation.py'
spec = importlib.util.spec_from_file_location('e1', SCRIPT)
e1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e1)


class E1Tests(unittest.TestCase):
    def test_known_weighted_confusion_and_kappa(self):
        m=e1.metrics([('sad','sad',2),('sad','fear',1),('fear','fear',1)],['sad','fear','angry'])
        self.assertAlmostEqual(m['accuracy'],.75)
        self.assertAlmostEqual(m['per_class']['sad']['f1'],.8)
        self.assertAlmostEqual(m['per_class']['fear']['f1'],2/3)
        self.assertAlmostEqual(m['macro_f1_fixed'],(.8+2/3)/3)
        self.assertAlmostEqual(e1.kappa(['a','a','b','b'],['a','b','a','b'])['cohen_kappa'],0)

        self.assertEqual(e1.metrics([('sad','angry',1)],['sad','angry'])['accuracy'],0)

    def test_group_sampling_and_probabilities(self):
        rows=[]
        for i in range(60):
            for source in ['a','b']:
                rows.append(dict(group_id=str(i),source=source,source_index=i,
                                 eligible_turns=i%4+1,predicted_core='contempt' if i%9==0 else 'fear'))
        frame=e1.select_frame(rows,9)
        self.assertEqual(len(frame),60)
        self.assertEqual(e1.select_frame(list(reversed(rows)),9),frame)
        sample,summary=e1.select_samples(frame,9,10,4)
        self.assertEqual(len(sample),len({x['group_id'] for x in sample}))
        self.assertEqual(e1.select_samples(list(reversed(frame)),9,10,4)[0],sample)
        for r in sample:
            if r['cohort']=='natural':
                self.assertAlmostEqual(r['group_inclusion_probability']*r['group_weight'],1)
                self.assertAlmostEqual(r['turn_weight'],r['group_weight']*r['eligible_turns'])
            else:
                self.assertIsNone(r['group_weight'])

    def test_no_future_or_private_annotation_in_target(self):
        row=dict(source='x',source_index=0,sample_id='abc',target_position=1,predicted_fine='sad')
        turns=[dict(role='supporter',text='prior',annotation={'emotion':'secret'}),
               dict(role='seeker',text='target'),dict(role='supporter',text='future secret')]
        tasks=e1.public_tasks([row],{('x',0):dict(turns=turns)})
        target=tasks[0]
        self.assertEqual([t['text'] for t in target['context']],['prior','target'])
        text=json.dumps(target)
        self.assertNotIn('predicted',text)
        self.assertNotIn('annotation',text)
        self.assertNotIn('future',text)

    def test_html_corpus_cannot_escape_script(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'a.html'
            e1.render_html(path,[dict(task_id='x',scope='target',context=[dict(role='seeker',text='OPTIONS </script><script>alert(1)</script>')])],'p')
            text=path.read_text(encoding='utf-8')
            self.assertNotIn('</script><script>alert',text)
            self.assertIn('\\u003c/script>',text)
            self.assertIn('OPTIONS \\u003c/script>',text)
            self.assertIn("data.package_id+'-'+location.pathname",text)

    def test_incomplete_same_person_and_adjudication_gates(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);bundle=root/'bundle';(bundle/'administrator').mkdir(parents=True)
            registry=dict(package_id='fixture',task_ids=['sample:target','sample:dialogue'])
            e1.write(bundle/'administrator/task_registry.json',registry)
            e1.write(bundle/'artifact_hashes.json',{})
            e1.write(bundle/'manifest.json',{'fixture':True})
            e1.write(bundle/'administrator/public_tasks.json',[dict(task_id=t,scope=t.split(':')[1],
                     context=[dict(role='seeker',text='synthetic')]) for t in registry['task_ids']])
            e1.write(bundle/'administrator/sample_key.json',[dict(sample_id='sample',source='fixture',
                cohort='natural',predicted_fine='fear',predicted_core='fear',group_weight=2,turn_weight=4)])
            def labels(name,values):
                path=root/(name+'.json')
                e1.write(path,dict(package_id='fixture',rater_id=name,annotations=[dict(task_id=t,
                    fine=v,core=v,language='en') for t,v in zip(registry['task_ids'],values)]))
                return path
            a=labels('A',['fear','fear']);b=labels('B',['sad','fear'])
            args=SimpleNamespace(bundle=bundle,rater_a=a,rater_b=b,adjudicated=None,output=root/'first')
            e1.score(args)
            self.assertFalse((args.output/'results.json').exists())
            self.assertEqual(e1.read(args.output/'status.json')['status'],'AWAITING_ADJUDICATION')
            c=labels('C',['sad'])
            args.adjudicated=c;args.output=root/'final';e1.score(args)
            result=e1.read(args.output/'results.json')
            self.assertEqual(result['metrics']['natural/all/target/sample']['core']['accuracy'],0)
            self.assertEqual(result['metrics']['natural/all/dialogue/sample']['core']['accuracy'],1)
            args.rater_b=a;args.output=root/'same'
            with self.assertRaisesRegex(ValueError,'distinct'):
                e1.score(args)
            incomplete=labels('B',[])
            args.rater_b=incomplete;args.output=root/'incomplete'
            with self.assertRaisesRegex(ValueError,'AWAITING'):
                e1.score(args)


if __name__=='__main__':
    unittest.main()
