"""Lazy model loading: run audio/video in their own migrated environments."""
import os
import sys
import tempfile
from pathlib import Path

from .core import ROOT, WEIGHTS, digest, file_hash, probe_audio, probe_video
from .references import verify_reference


def seed_all(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AudioWorker:
    def __init__(self, run, options):
        sys.path[:0] = [str(ROOT / 'MeloTTS-main'), str(ROOT / 'OpenVoice-main')]
        from openvoice.api import ToneColorConverter
        self.options = options
        self.run = Path(run)
        self.device = 'cuda:0'
        self.converter = ToneColorConverter(str(WEIGHTS / 'openvoice_weights/converter/config.json'),
                                            device=self.device)
        self.converter.load_ckpt(str(WEIGHTS / 'openvoice_weights/converter/checkpoint.pth'))
        self.converter_fingerprint = {
            name: dict(size=(WEIGHTS / 'openvoice_weights/converter' / name).stat().st_size,
                       mtime_ns=(WEIGHTS / 'openvoice_weights/converter' / name).stat().st_mtime_ns)
            for name in ('config.json', 'checkpoint.pth')}
        self.models = {}
        self.embeddings = {}
        self.cache = Path(run) / 'speaker_embeddings'
        self.cache.mkdir(exist_ok=True)

    def generate(self, item, output):
        import torch
        from melo.api import TTS
        from openvoice import se_extractor
        verify_reference(item)
        language = item['language']
        if language not in self.models:
            name = 'MeloTTS-Chinese' if language == 'ZH' else 'MeloTTS-English-v3'
            weights = ROOT / 'MeloTTS-main/myshell-ai' / name
            model = TTS(language=language, device=self.device, use_hf=False,
                        config_path=str(weights / 'config.json'), ckpt_path=str(weights / 'checkpoint.pth'))
            if len(model.hps.data.spk2id) != 1:
                raise ValueError('Select an explicit base speaker for a multi-speaker Melo model')
            speaker_key, speaker_id = next(iter(model.hps.data.spk2id.items()))
            se_path = WEIGHTS / 'openvoice_weights/base_speakers/ses' / (speaker_key.lower().replace('_', '-') + '.pth')
            source = torch.load(str(se_path), map_location=self.device)
            self.models[language] = model, speaker_id, source, str(se_path)
        model, speaker_id, source, se_path = self.models[language]
        ref = item['reference']
        if file_hash(ref['audio']) != ref['audio_sha256']:
            raise ValueError('Reference audio fingerprint changed')
        key = digest([ref['audio_sha256'], self.converter.version, self.options['reference_method'],
                      self.converter_fingerprint])
        cache = self.cache / (key + '.pth')
        if key not in self.embeddings:
            if cache.exists():
                target = torch.load(str(cache), map_location=self.device)
            elif self.options['reference_method'] == 'silero_vad':
                target, _ = se_extractor.get_se(ref['audio'], self.converter, target_dir=str(self.cache), vad=True)
                torch.save(target.cpu(), str(cache) + '.tmp')
                os.replace(str(cache) + '.tmp', cache)
            else:

                target = self.converter.extract_se(ref['audio'])
                torch.save(target.cpu(), str(cache) + '.tmp')
                os.replace(str(cache) + '.tmp', cache)
            if not torch.isfinite(target).all() or target.shape != source.shape:
                raise ValueError('Invalid/mismatched speaker embeddings')
            self.embeddings[key] = target.to(self.device)
        seed_all(item['seed'])
        with tempfile.TemporaryDirectory(prefix='tts_', dir=str(output.parent)) as tmp:
            base = Path(tmp) / 'melo.wav'
            converted = Path(tmp) / 'converted.wav'
            tts_options = {key: self.options[key] for key in
                           ('speed', 'sdp_ratio', 'noise_scale', 'noise_scale_w')}
            model.tts_to_file(item['text'], speaker_id, str(base), **tts_options, quiet=True)
            base_qc = probe_audio(base)
            self.converter.convert(str(base), src_se=source, tgt_se=self.embeddings[key],
                                   output_path=str(converted), tau=self.options['tau'], message='@MyShell')
            qc = probe_audio(converted)
            if abs(qc['seconds'] - base_qc['seconds']) > .25:
                raise ValueError('Voice conversion changed audio duration')
            base_output = self.run / 'audio_base' / (item['asset_id'] + '.wav')
            base_output.parent.mkdir(exist_ok=True)
            os.replace(base, base_output)
            os.replace(converted, output)
        return dict(qc=qc, base_qc=base_qc, base_speaker_embedding=se_path,
                    base_audio=str(base_output), base_audio_sha256=file_hash(base_output),
                    tts_options=tts_options, reference_audio=ref['audio'],
                    reference_audio_sha256=ref['audio_sha256'],
                    intended_speech_emotion=item['emotion'],
                    reference_method=self.options['reference_method'], watermark=False,
                    speech_emotion_control=('base_tts_default_with_identity_conversion' if
                        item['emotion'] == 'neutral' else 'avamerg_emotion_matched_reference; '
                        'transfer_strength_requires_perceptual_evaluation'))


class VideoWorker:
    def __init__(self, run, options):
        sys.path.insert(0, str(ROOT / 'DICE-Talk-main'))
        from dice_talk import DICE_Talk
        self.model = DICE_Talk(device_id=0, enable_interpolate_frame=True)
        self.options = options
        self.run = Path(run)
        self.portraits = {}

    def generate(self, item, output, audio_path=None):
        import cv2
        import numpy as np
        verify_reference(item)
        ref = item['reference']
        if file_hash(ref['portrait']) != ref['portrait_sha256']:
            raise ValueError('Reference portrait fingerprint changed')
        key = ref['portrait_sha256']
        if key not in self.portraits:
            portrait = self.run / 'references' / (key[:20] + '_crop.png')
            check = self.model.preprocess(ref['portrait'], expand_ratio=.35)
            if check['face_num'] != 1:
                raise ValueError(f"Reference requires exactly one face; found {check['face_num']}")
            self.model.crop_image(ref['portrait'], str(portrait), check['crop_bbox'])
            image = cv2.imread(str(portrait))
            if image is None or min(image.shape[:2]) < 128:
                raise ValueError('Cropped face resolution is too small')
            self.portraits[key] = str(portrait)
        audio = Path(audio_path) if audio_path is not None else self.run / 'audio' / (item['asset_id'] + '.wav')
        aq = probe_audio(audio)
        segment_seconds = self.options.get('video_segment_seconds', 0)
        if not segment_seconds and aq['seconds'] > self.options['max_video_seconds']:
            raise ValueError('Utterance exceeds configured video duration; segment explicitly, never truncate')
        prior = ROOT / 'DICE-Talk-main/examples/emo' / (item['emotion'] + '.npy')
        np.load(str(prior), allow_pickle=True).item()['mu']
        if segment_seconds and aq['seconds'] > segment_seconds:
            from .long_video import segmented_video
            details = segmented_video(self.model, audio, self.portraits[key], prior, output,
                                      self.run / 'video_parts' / item['asset_id'], self.options, item['seed'])
            return dict(details, cropped_portrait=self.portraits[key],
                        intended_face_emotion=item['emotion'], emotion_prior_sha256=file_hash(prior))
        tmp = output.with_name(output.stem + '.partial.mp4')
        try:
            result = self.model.process(self.portraits[key], str(audio), str(prior), str(tmp),
                                        min_resolution=512, inference_steps=self.options['steps'],
                                        ref_scale=self.options['ref_scale'], emo_scale=self.options['emo_scale'],
                                        seed=item['seed'])
            if result != 0:
                raise RuntimeError(f'DICE generation failed with status {result}')
            qc = probe_video(tmp, aq['seconds'], decode=True)
            os.replace(tmp, output)
        finally:
            if tmp.exists():
                tmp.unlink()
        return dict(qc=qc, cropped_portrait=self.portraits[key],
                    intended_face_emotion=item['emotion'], emotion_prior_sha256=file_hash(prior))
