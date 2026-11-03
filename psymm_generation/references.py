"""Build a filename catalog from AvaMERG metadata; inspect only chosen references."""
import random
from pathlib import Path

from .core import atomic_json, checked, digest, emotion, file_hash, probe_audio


def catalog_from_metadata(records, avamerg):
    catalog = {}
    for record in records:
        cid = str(record['conversation_id'])
        profiles = {role: str(record[role + '_profile']['ID']) for role in ('speaker', 'listener')}

        history = {int(u['index']): u for t in record['turns'] for u in t['dialogue_history']}
        for index, u in history.items():
            person = profiles[u['role']]
            stem = f'dia{cid}utt{index}_{person}'
            ref = dict(identity=person, segment=stem,
                       audio=str(avamerg / 'audio_v5_0/audio_v5_0' / (stem + '.wav')),
                       video=str(avamerg / 'video_v5_0' / (stem + '.mp4')),
                       source_dialogue=cid, source_turn=index, source_role=u['role'],
                       reference_text=u['utterance'])
            catalog.setdefault(person, {})[stem] = ref
    return {k: sorted(v.values(), key=lambda x: x['segment']) for k, v in catalog.items()}


def emotion_pool_from_filenames(catalog, directory, audit=None):
    """Index pool membership by filename only; never open the dataset's media."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f'AvaMERG emotion pool missing: {directory}')
    by_segment = {r['segment']: r for refs in catalog.values() for r in refs}
    pool, membership, entries = {}, {}, []
    for folder in sorted(directory.iterdir()):
        if not folder.is_dir():
            continue
        label = emotion(folder.name)
        if label == 'neutral':
            continue
        for path in sorted(folder.glob('*.wav')):
            if path.stem not in by_segment:
                raise ValueError(f'Emotion reference absent from identity metadata: {path.name}')
            membership.setdefault(path.stem, set()).add(label)
            entries.append((path, label))
    ambiguous = {stem: sorted(labels) for stem, labels in membership.items() if len(labels) > 1}
    for path, label in entries:
        if path.stem in ambiguous:
            continue
        original = by_segment[path.stem]
        ref = dict(original, audio=str(path.resolve()), reference_emotion=label,
                   reference_source='avamerg_emotion_pool')
        pool.setdefault(label, {}).setdefault(ref['identity'], {})[ref['segment']] = ref
    if audit is not None:
        audit.update(excluded_ambiguous_references=ambiguous,
                     eligible_references={label: sum(len(refs) for refs in people.values())
                                          for label, people in pool.items()},
                     policy='Filename/metadata index only; exclude conflicting pool memberships; '
                            'neutral is never sampled from an emotion pool')
    return {label: {person: sorted(refs.values(), key=lambda r: r['segment'])
                    for person, refs in people.items()} for label, people in pool.items()}


def identity_pair(catalog, dialogue_id, seed, eligible=None):
    identities = sorted(catalog)
    if len(identities) < 2:
        raise ValueError('At least two distinct reference identities are required')
    rng = random.Random(int(digest([seed, dialogue_id])[:16], 16))
    if eligible is None:
        return dict(zip(('seeker', 'supporter'), rng.sample(identities, 2)))
    seekers = sorted(p for p in eligible['seeker']
                     if any(other != p for other in eligible['supporter']))
    if not seekers:
        raise ValueError('Emotion pools cannot provide two distinct dialogue identities')
    seeker = rng.choice(seekers)
    supporter = rng.choice(sorted(p for p in eligible['supporter'] if p != seeker))
    return dict(seeker=seeker, supporter=supporter)


def validate_reference_speech(path):
    import sys
    import librosa
    import torch
    from .core import ROOT
    sys.path.insert(0, str(ROOT / 'OpenVoice-main'))
    from openvoice.se_extractor import get_vad_segments
    samples, _ = librosa.load(str(path), sr=16000, mono=True)
    segments = get_vad_segments(torch.from_numpy(samples), output_sample=True,
                                min_speech_duration=.1, min_silence_duration=1,
                                method='silero')
    seconds = sum(int(segment['end'] * 1000 / 16000) -
                  int(segment['start'] * 1000 / 16000) for segment in segments) / 1000
    if seconds < 1.5:
        raise ValueError(f'Reference VAD speech is too short: {seconds:.3f}s')
    return seconds


def select_anchor(catalog, person, cache, seed, scope=None, target_emotion='neutral',
                  validate_speech=False):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    candidates = list(catalog[person])
    selection = digest([seed, scope, person, target_emotion,
                        [(r['segment'], r['audio']) for r in candidates]])
    saved = cache / f'anchor_{selection[:24]}.json'
    if saved.exists():
        import json
        ref = json.loads(saved.read_text(encoding='utf-8'))
        for key in ('audio', 'portrait'):
            if file_hash(ref[key]) != ref[key + '_sha256']:
                raise ValueError(f'Reference changed: {ref[key]}')
        if validate_speech:
            validate_reference_speech(ref['audio'])
        return ref
    random.Random(int(selection[:16], 16)).shuffle(candidates)
    failures = []
    for ref in candidates[:40]:
        try:
            q = probe_audio(ref['audio'])
            if q['seconds'] < 1.5 or q['silence_fraction'] > .65:
                continue
            if validate_speech:
                q['vad_speech_seconds'] = validate_reference_speech(ref['audio'])
            portrait = cache / (ref['segment'] + '.png')
            checked(['ffmpeg', '-v', 'error', '-y', '-threads', '1',
                     '-ss', str(q['seconds'] / 2), '-i', ref['video'],
                     '-threads', '1', '-filter_threads', '1', '-frames:v', '1', portrait],
                    timeout=60)
            if not portrait.exists() or portrait.stat().st_size < 1000:
                raise ValueError('Reference portrait extraction failed')
            ref = dict(ref, portrait=str(portrait.resolve()), audio_qc=q,
                       reference_emotion=ref.get('reference_emotion'),
                       reference_source=ref.get('reference_source', 'avamerg_identity_anchor'),
                       dialogue_scope=scope, intended_emotion=target_emotion,
                       pairing='same AvaMERG dialogue/turn/profile ID; middle video frame',
                       audio_sha256=file_hash(ref['audio']), portrait_sha256=file_hash(portrait))
            atomic_json(saved, ref)
            return ref
        except (OSError, ValueError, RuntimeError) as e:
            failures.append(str(e))
        except Exception as e:

            failures.append(repr(e))
    raise ValueError(f'No usable paired reference for identity {person}: {failures[-3:]}')


def select_dialogue_references(catalog, pool, dialogue, cache, seed, validate_speech=False):
    """One paired audio/portrait per role, fixed for this entire dialogue."""
    role_emotions = {role: {t['emotion'] for t in dialogue['turns'] if t['role'] == role}
                     for role in ('seeker', 'supporter')}
    if any(len(labels) > 1 for labels in role_emotions.values()):
        raise ValueError('Multiple emotions within one role conflict with the fixed audio anchor; '
                         'an explicit policy is required, never silently switch identities/references')
    labels = {role: next(iter(values), 'neutral') for role, values in role_emotions.items()}
    choices = {}
    for role, label in labels.items():
        choices[role] = catalog if label == 'neutral' else pool.get(label, {})
        if not choices[role]:
            raise ValueError(f'No AvaMERG references for {role} emotion {label}; no neutral fallback')
    pair = identity_pair(catalog, dialogue['dialogue_id'], seed,
                         {role: sorted(refs) for role, refs in choices.items()})


    orders = {}
    for role in ('seeker', 'supporter'):
        others = sorted(p for p in choices[role] if p != pair[role])
        random.Random(int(digest([seed, dialogue['dialogue_id'], role])[:16], 16)).shuffle(others)
        orders[role] = [pair[role], *others]
    failures = []
    for seeker in orders['seeker']:
        if not any(p != seeker for p in orders['supporter']):
            continue
        try:
            first = select_anchor(choices['seeker'], seeker, cache, seed,
                                  scope=f"{dialogue['dialogue_id']}:seeker",
                                  target_emotion=labels['seeker'], validate_speech=validate_speech)
        except ValueError as error:
            failures.append(str(error))
            continue
        for supporter in orders['supporter']:
            if supporter == seeker:
                continue
            try:
                second = select_anchor(choices['supporter'], supporter, cache, seed,
                                       scope=f"{dialogue['dialogue_id']}:supporter",
                                       target_emotion=labels['supporter'], validate_speech=validate_speech)
                return dict(seeker=first, supporter=second)
            except ValueError as error:
                failures.append(str(error))
    raise ValueError(f'No usable emotion-matched pair for {dialogue["dialogue_id"]}: {failures[-3:]}')


def verify_reference(item):
    """Check actual selected files and label agreement before inference or reuse."""
    ref = item['reference']
    if item['emotion'] == 'neutral':
        if ref.get('reference_emotion') is not None:
            raise ValueError('Neutral must not select an emotional pool reference')
    elif ref.get('reference_emotion') != item['emotion']:
        raise ValueError('Speech reference emotion does not match the planned face emotion')
    for key in ('audio', 'portrait'):
        if file_hash(ref[key]) != ref[key + '_sha256']:
            raise ValueError(f'Reference {key} fingerprint changed')
