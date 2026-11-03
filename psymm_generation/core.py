import hashlib
import json
import os
import re
import subprocess
import unicodedata
from pathlib import Path

ROOT = Path(os.environ.get('PSYMM_ROOT', Path(__file__).resolve().parents[1]))
WEIGHTS = Path(os.environ.get('PSYMM_WEIGHTS', ROOT / 'weights'))
DATA = Path(os.environ.get('PSYMM_DATA', ROOT / 'data'))
INPUTS = {
    'esconv': 'infer_outputs_esconv/esconv_emo.json',
    'extes': 'infer_outputs_extes/train_grouped_annotated.json',
    'openr1': 'infer_outputs_openr1-psy/train_annotation.json',
    'psych8k': 'infer_outputs_psy8k/Alexander_Street_shareGPT_2.0_ID_annotation.json',
    'psydial': 'infer_outputs_psydial/PsyDial-D1_annotation.json',
    'psyqa': 'infer_outputs_psyqa/PsyQA_full_annotation.json',
}
EMOTIONS = {
    'neutral': 'neutral', 'none': 'neutral',
    'anger': 'angry', 'angry': 'angry', 'annoyance': 'angry',
    'fear': 'fear', 'anxiety': 'fear', 'anxious': 'fear', 'panic': 'fear', 'paranoia': 'fear',
    'sad': 'sad', 'sadness': 'sad', 'depression': 'sad', 'depressed': 'sad',
    'emptiness': 'sad', 'hopelessness': 'sad', 'loneliness': 'sad', 'grief': 'sad',
    'shame': 'sad', 'guilt': 'sad', 'pain': 'sad', 'nervousness': 'fear',
    'jealousy': 'angry',
    'happy': 'happy', 'happiness': 'happy', 'joy': 'happy',
    'surprise': 'surprised', 'surprised': 'surprised',
    'disgust': 'disgusted', 'disgusted': 'disgusted', 'contempt': 'contempt',
}
ROLES = {'seeker': 'seeker', 'speaker': 'seeker', 'user': 'seeker', 'human': 'seeker',
         'patient': 'seeker', 'supporter': 'supporter', 'listener': 'supporter',
         'assistant': 'supporter', 'gpt': 'supporter', 'counselor': 'supporter'}


SUPPORTER_EMOTIONS = {'extes': 'happy', 'openr1': 'happy', 'psych8k': 'sad',
                     'psydial': 'happy', 'psyqa': 'happy'}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def clean_text(text):
    if not isinstance(text, str):
        raise ValueError('Text must be a string')

    text = ''.join(' ' if unicodedata.category(c) in ('Cc', 'Cf') else c for c in text)
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        raise ValueError('Empty utterance')
    return text


def emotion(value):
    key = str(value).strip().lower()
    if key not in EMOTIONS:
        raise ValueError(f'Unmapped emotion {value!r}; add an explicit documented mapping')
    return EMOTIONS[key]


def normalized_dialogue(source, record, index, qa_answers='first'):
    """Return independent dialogue units; never speak instructions or hidden reasoning."""
    source_id = next((record[k] for k in ('conversation_id', 'dialogueID', 'post_id', 'questionID')
                      if k in record), index)
    dialogue_emotion = record.get('emotion_type')
    if source == 'psyqa':
        answers = record.get('answers', [])
        if qa_answers == 'first':
            answers = answers[:1]
        if not answers:
            raise ValueError(f'PsyQA question {source_id} has no answers')
        groups = [[{'speaker': 'seeker', 'content': '\n'.join(filter(None, [record.get('question'), record.get('description')]))},
                   {'speaker': 'supporter', 'content': a['answer_text']}]
                  for a in answers]
    elif source == 'openr1':
        groups = [[t for r in record['conversation'] for t in
                   ({'speaker': 'seeker', 'content': r['patient']},
                    {'speaker': 'supporter', 'content': r['counselor_content']})]]
    elif source == 'psych8k':
        groups = [[{'speaker': 'seeker', 'content': record['input']},
                   {'speaker': 'supporter', 'content': record['output']}]]
    else:
        groups = [record.get('dialog', record.get('messages', []))]
    for answer_index, group in enumerate(groups):
        turns = []
        for raw_index, turn in enumerate(group):
            raw_role = turn.get('speaker', turn.get('role', turn.get('from')))
            if raw_role == 'system':
                continue
            if raw_role not in ROLES:
                raise ValueError(f'Unknown role {raw_role!r} in {source_id}')
            role = ROLES[raw_role]
            annotation = turn.get('annotation', {}) or {}
            explicit = turn.get('emotion_type', turn.get('emotion', annotation.get('emotion')))
            if explicit is not None:
                raw_emotion, origin = explicit, 'turn_annotation'
            elif role == 'seeker' and dialogue_emotion is not None:
                raw_emotion, origin = dialogue_emotion, 'dialogue_annotation'
            elif role == 'supporter' and source in SUPPORTER_EMOTIONS:
                raw_emotion, origin = SUPPORTER_EMOTIONS[source], 'original_source_supporter_policy'
            else:
                raw_emotion, origin = 'neutral', 'unannotated_neutral_policy'
            raw_text = turn.get('content', turn.get('text', turn.get('value')))
            text = clean_text(raw_text)
            turns.append(dict(turn_index=raw_index, role=role, original_role=raw_role,
                              original_text=raw_text, text=text,
                              language='ZH' if re.search('[\u3400-\u9fff]', text) else 'EN_NEWEST',
                              emotion=emotion(raw_emotion), raw_emotion=raw_emotion,
                              emotion_origin=origin, annotation=annotation))
        if not turns:
            raise ValueError(f'No dialogue turns in {source_id}')
        yield dict(source=source, source_id=source_id, source_index=index,
                   answer_index=answer_index if source == 'psyqa' else None,
                   dialogue_id=f'{source}_{index:06d}_{digest(source_id)[:8]}' +
                   (f'_a{answer_index:02d}' if source == 'psyqa' else ''), turns=turns)


def checked(command, timeout=None):
    return subprocess.run([str(x) for x in command], check=True, capture_output=True,
                          text=True, timeout=timeout)


def probe_video(path, expected_duration=None, decode=False):
    info = json.loads(checked(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                               '-of', 'json', path]).stdout)
    streams = info['streams']
    video = next(s for s in streams if s['codec_type'] == 'video')
    audio = next(s for s in streams if s['codec_type'] == 'audio')
    vd = float(video.get('duration', info['format']['duration']))
    ad = float(audio.get('duration', info['format']['duration']))
    if min(vd, ad) <= 0 or abs(vd - ad) > .25:
        raise ValueError(f'Invalid A/V durations {vd}/{ad}')
    if expected_duration is not None and max(abs(vd - expected_duration), abs(ad - expected_duration)) > .25:
        raise ValueError(f'Truncated/extended output: {vd}/{ad}, expected {expected_duration}')
    if decode:
        checked(['ffmpeg', '-v', 'error', '-xerror', '-i', path, '-f', 'null', '-'])
    return dict(video_seconds=vd, audio_seconds=ad, width=video['width'], height=video['height'],
                fps=video['avg_frame_rate'], frames=video.get('nb_frames'), full_decode=decode)


def probe_audio(path):
    import numpy as np
    import soundfile as sf
    wav, sr = sf.read(str(path), dtype='float32', always_2d=True)
    if not len(wav) or not np.isfinite(wav).all():
        raise ValueError('Empty or non-finite audio')
    rms = float(np.sqrt(np.mean(wav ** 2)))
    clip = float(np.mean(np.abs(wav) >= .999))
    if len(wav) / sr < .25 or rms < .001 or clip > .01:
        raise ValueError(f'Audio quality gate failed: seconds={len(wav)/sr}, rms={rms}, clipping={clip}')
    return dict(seconds=len(wav)/sr, sample_rate=sr, channels=wav.shape[1],
                rms=rms, peak=float(np.max(np.abs(wav))), clipping_fraction=clip,
                silence_fraction=float(np.mean(np.max(np.abs(wav), axis=1) < .001)))
