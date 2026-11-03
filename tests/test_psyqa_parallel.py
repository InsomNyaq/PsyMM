import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.psyqa_parallel import (partitions, gpu_list, reference_maps, launch,
                                        approve_long_video)


class ParallelContract(unittest.TestCase):
    def test_shards_cover_questions_exactly_and_keep_global_indices(self):
        for size in (1, 7, 22341):
            for workers in (1, min(4,size), min(8,size)):
                spans = partitions(size, workers)
                indices = [i for s in spans for i in range(s['offset'], s['offset']+s['limit'])]
                self.assertEqual(indices, list(range(size)))
        with self.assertRaises(ValueError):
            partitions(2,3)

    def test_gpu_assignment(self):
        self.assertEqual(gpu_list('1,3',2), [1,1,3,3])
        for ids in ('0,0', '0;touch bad', '-1', ''):
            with self.assertRaises(ValueError):
                gpu_list(ids,1)

    def test_maps_retain_identity_source_and_answer_and_are_planned_not_completed(self):
        ref = dict(identity='42', segment='dia1utt0_42', reference_source='avamerg_emotion_pool',
                   reference_emotion='sad', audio='/dia1utt0_42.wav', audio_sha256='a',
                   portrait='/dia1utt0_42.png', portrait_sha256='b', video='/dia1utt0_42.mp4',
                   dialogue_scope='psyqa_7_a01:seeker')
        item = dict(source_index=7, source_id=123, dialogue_id='psyqa_7_a01', answer_index=1,
                    role='seeker', emotion='sad', emotion_origin='dialogue_annotation',
                    source='psyqa', asset_id='psyqa_7_a01_T0000', reference=ref)
        second = copy.deepcopy(item)
        second.update(role='supporter', asset_id='psyqa_7_a01_T0001')
        second['reference'].update(identity='43', segment='dia1utt1_43',
            audio='/dia1utt1_43.wav', portrait='/dia1utt1_43.png', video='/dia1utt1_43.mp4',
            dialogue_scope='psyqa_7_a01:supporter')
        with tempfile.TemporaryDirectory() as tmp, patch('experiments.psyqa_parallel.read_plan',return_value=[item,second]):
            run = Path(tmp)
            reference_maps(run)
            a=json.loads((run/'audio_reference_map_emo.json').read_text())
            v=json.loads((run/'ref_map.json').read_text())
            self.assertEqual(a['records'][item['asset_id']]['answer_index'],1)
            self.assertEqual(v['records'][item['asset_id']]['identity'],'42')
            self.assertEqual(v['records'][item['asset_id']]['reference_video'],'/dia1utt0_42.mp4')
            self.assertIn('not proof',a['note'])

    def test_dry_run_has_no_outputs_and_launch_is_blocked_until_pilot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'source.json').write_text(json.dumps([dict(answers=[{},{}]),dict(answers=[{}])]))
            args=argparse.Namespace(session='test',gpus='0',instances_per_gpu=1,run_dir=root/'out',
                                    qa_answers='all',seed=17,video_segment_seconds=20,dry_run=True)
            with patch('experiments.psyqa_parallel.ROOT',root), \
                 patch('experiments.psyqa_parallel.INPUTS',{'psyqa':'source.json'}), \
                 patch('experiments.psyqa_parallel.subprocess.run') as process:
                self.assertEqual(launch(args),0)
                self.assertFalse(args.run_dir.exists())
                process.assert_not_called()
                args.dry_run=False
                with self.assertRaisesRegex(ValueError,'pilot'):
                    launch(args)
                process.assert_not_called()

    def test_approval_requires_validated_segmented_pilot_and_existing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pilot = root/'pilot'; pilot.mkdir()
            (pilot/'validation.json').write_text(json.dumps(dict(failed_checks=0, records=[{}])))
            (pilot/'video.config.json').write_text(json.dumps(dict(options=dict(video_segment_seconds=20))))
            import sqlite3
            with sqlite3.connect(str(pilot/'state.sqlite')) as db:
                db.execute('CREATE TABLE media (stage TEXT, status TEXT, details TEXT)')
                db.execute('INSERT INTO media VALUES (?,?,?)', ('video','complete',json.dumps(dict(segmented=True))))
            evidence = root/'review.txt'; evidence.write_text('Reviewed every join.')
            args = argparse.Namespace(pilot_run=pilot, video_segment_seconds=20,
                                      reviewer='reviewer', evidence=evidence)
            with patch('experiments.psyqa_parallel.ROOT', root), \
                 patch('psymm_generation.cli.inventory', return_value=dict(code={'x':'y'})):
                self.assertEqual(approve_long_video(args), 0)
                approval = json.loads((root/'experiments/psyqa_long_video_gpu_approval.json').read_text())
            self.assertEqual(approval['reviewer'], 'reviewer')
            self.assertEqual(approval['completed_segmented_videos'], 1)

    def test_explicit_skip_launches_and_records_unreviewed_without_forging_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'source.json').write_text(json.dumps([dict(answers=[{}, {}])]))
            args = argparse.Namespace(session='test', gpus='3', instances_per_gpu=1,
                                      run_dir=root/'out', qa_answers='all', seed=17,
                                      video_segment_seconds=20, dry_run=False,
                                      skip_visual_review=True)
            with patch('experiments.psyqa_parallel.ROOT', root), \
                 patch('experiments.psyqa_parallel.INPUTS', {'psyqa':'source.json'}), \
                 patch('experiments.psyqa_parallel.subprocess.run') as process:
                process.return_value.returncode = 1
                self.assertEqual(launch(args), 0)
            receipt = json.loads((args.run_dir/'last_launch.json').read_text())
            self.assertEqual(receipt['visual_review'], 'skipped_by_user_request')
            self.assertEqual(receipt['gpu_per_instance'], [3])
            self.assertFalse((root/'experiments/psyqa_long_video_gpu_approval.json').exists())
            commands = [call.args[0] for call in process.call_args_list]
            self.assertTrue(any(cmd[:2] == ['tmux', 'new-window'] for cmd in commands))


if __name__=='__main__':
    unittest.main()
