"""CPU ffmpeg integration; substitutes synthetic video for DICE, never loads torch."""
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from psymm_generation.core import checked, file_hash, probe_video
from psymm_generation.long_video import windows, segmented_video, video_info


class SyntheticDice:
    def __init__(self, fail_call=None):
        self.calls=[]
        self.fail_call=fail_call

    def process(self, portrait, audio, prior, output, **kwargs):
        self.calls.append((portrait,prior,kwargs))
        if len(self.calls)==self.fail_call:
            raise RuntimeError('simulated interruption')
        d=sf.info(audio).duration
        checked(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=c=blue:s=64x64:r=25',
                 '-i',audio,'-map','0:v:0','-map','1:a:0','-t',str(d),
                 '-c:v','libx264','-pix_fmt','yuv420p','-threads','1','-c:a','aac',output])
        return 0


class WindowContract(unittest.TestCase):
    def test_sample_coverage_including_short_tail(self):
        for samples in (1,32000,64000,64208,32001):
            parts=windows(samples,16000,2)
            self.assertEqual(parts[0]['start'],0)
            self.assertEqual(parts[-1]['end'],samples)
            self.assertEqual(sum(p['end']-p['start'] for p in parts),samples)
            for a,b in zip(parts,parts[1:]):
                self.assertEqual(a['end'],b['start'])
            for p in parts:
                self.assertLessEqual(p['context_start'],p['start'])
                self.assertGreaterEqual(p['context_end'],p['end'])
            self.assertEqual(parts[-1]['right_padding_samples'],16000)
            self.assertTrue(all(p['right_padding_samples'] == 0 for p in parts[:-1]))
            self.assertEqual(sum(math.ceil((p['end']-p['start'])*25/16000) for p in parts),
                             math.ceil(samples*25/16000))


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'ffmpeg required')
class LongVideoIntegration(unittest.TestCase):
    def test_interrupt_resume_corruption_and_complete_audio_timeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            audio=root/'full.wav'
            data=.1*np.sin(2*np.pi*440*np.arange(64208)/16000)
            sf.write(str(audio),data,16000,subtype='PCM_16')
            original_hash=file_hash(audio)
            portrait=root/'portrait.png'; portrait.write_bytes(b'portrait fixture')
            prior=root/'sad.npy'; prior.write_bytes(b'prior fixture')
            options=dict(video_segment_seconds=2,steps=25,ref_scale=3.,emo_scale=6.,_configuration_hash='fixture')
            args=(audio,portrait,prior,root/'full.mp4',root/'cache',options,17)
            failed=SyntheticDice(fail_call=2)
            with self.assertRaisesRegex(RuntimeError,'interruption'):
                segmented_video(failed,*args)
            self.assertFalse((root/'full.mp4').exists())
            resumed=SyntheticDice()
            result=segmented_video(resumed,*args)
            self.assertEqual(len(resumed.calls),2)
            self.assertEqual(len(result['segments']),3)
            self.assertEqual(result['segments'][-1]['part']['right_padding_samples'],16000)
            self.assertEqual(file_hash(audio),original_hash)
            self.assertEqual(int(video_info(root/'full.mp4')['nb_frames']),101)
            probe_video(root/'full.mp4',64208/16000,True)
            for call in resumed.calls:
                self.assertEqual(call[0],str(portrait))
                self.assertEqual(call[1],str(prior))

            decoded=root/'decoded.wav'
            checked(['ffmpeg','-v','error','-y','-i',root/'full.mp4','-vn','-acodec','pcm_s16le',decoded])
            actual,rate=sf.read(str(decoded))
            self.assertEqual(rate,16000)
            for start in (0,31000,63000):
                end=min(start+1000,len(data))
                self.assertGreater(np.corrcoef(data[start:end],actual[start:end])[0,1],.98)
            again=SyntheticDice()
            segmented_video(again,*args)
            self.assertEqual(len(again.calls),0)
            Path(result['segments'][1]['path']).write_bytes(b'corrupted')
            repaired=SyntheticDice()
            segmented_video(repaired,*args)
            self.assertEqual(len(repaired.calls),1)
            self.assertEqual(file_hash(audio),original_hash)


if __name__=='__main__':
    unittest.main()
