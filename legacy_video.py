#!/usr/bin/env python3
"""Generate video from existing WAVs and their original reference map.

Planning reads JSON metadata and directory entry names only. Existing audio
must already use two distinct fixed role identities for each whole dialogue.
"""
import argparse
import contextlib
import json
import multiprocessing as mp
import os
import signal
import sqlite3
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from psymm_generation.core import atomic_json, digest, file_hash, probe_video


def identity_audit(records):
    """Audit the whole audio map, including missing media and turns beyond --limit."""
    dialogues, invalid = {}, []
    roles = {'listener': 'supporter', 'speaker': 'seeker',
             'supporter': 'supporter', 'seeker': 'seeker',
             'user': 'seeker', 'assistant': 'supporter'}
    for name, entry in records.items():
        cid = entry.get('post_id', entry.get('conversation', entry.get('dialogue_id')))
        role = roles.get(entry.get('role'))
        reference = Path(entry.get('reference_audio', '')).stem
        if cid is None or role is None or '_' not in reference or not reference.rsplit('_', 1)[-1]:
            invalid.append(name)
            continue
        if entry.get('reference_path') and Path(entry['reference_path']).stem != reference:
            invalid.append(name)
            continue
        dialogues.setdefault(str(cid), {}).setdefault(role, set()).add(reference.rsplit('_', 1)[-1])
    changed, shared, incomplete = [], [], []
    for cid, bindings in dialogues.items():
        if any(len(ids) > 1 for ids in bindings.values()):
            changed.append(cid)
        if bindings.get('seeker', set()) & bindings.get('supporter', set()):
            shared.append(cid)
        if set(bindings) != {'seeker', 'supporter'}:
            incomplete.append(cid)
    return dict(dialogues_with_identity_changes=len(changed),
                dialogues_with_shared_role_identity=len(shared),
                dialogues_missing_role=len(incomplete), invalid_identity_records=len(invalid),
                identity_issue_examples=dict(changed=changed[:10], shared=shared[:10],
                                             incomplete=incomplete[:10], invalid=invalid[:10]))


def require_fixed_identities(summary):
    if any(summary[key] for key in ('dialogues_with_identity_changes',
            'dialogues_with_shared_role_identity', 'dialogues_missing_role', 'invalid_identity_records')):
        raise ValueError('Existing audio map violates the fixed two-role dialogue identity policy. '
                         'Video cannot repair changing audio identities. Preserve old files and prepare '
                         'a NEW run, then regenerate audio and video with run_multimodal.sh. '
                         'No model loaded or existing media/map modified.')


def plan(args):
    directory = args.dataset_dir.resolve()
    mapping = args.audio_map or directory / 'audio_reference_map_emo.json'
    records = json.loads(mapping.read_text(encoding='utf-8'))
    if not isinstance(records, dict) or not records:
        raise ValueError('Audio reference map must be a nonempty filename -> reference dictionary')
    audio_dir = args.audio_dir or directory / 'OpenVoiceEMO_outputs'
    image_dir = args.image_dir or ROOT / 'infer_outputs_esconv/avamerg_test_images'

    audio_names = {p.name for p in os.scandir(audio_dir) if p.name.endswith('.wav')}
    image_names = {p.name for p in os.scandir(image_dir)}
    audit = identity_audit(records)
    items, missing = [], Counter()
    aliases = {'surprise': 'surprised', 'disgust': 'disgusted', 'anger': 'angry',
               'sadness': 'sad', 'happiness': 'happy'}
    for name, entry in sorted(records.items()):
        if Path(name).name != name or not name.endswith('.wav'):
            raise ValueError(f'Invalid audio filename: {name!r}')
        if name not in audio_names:
            missing['missing_audio'] += 1
            continue
        reference = Path(entry['reference_audio']).stem
        face = next((reference + ext for ext in ('.jpg', '.png', '.jpeg')
                     if reference + ext in image_names), None)
        if face is None:
            missing['missing_portrait'] += 1
            continue
        label = str(entry.get('emotion', 'neutral')).strip().lower()
        label = aliases.get(label, label)
        if label not in ('fear', 'sad', 'angry', 'happy', 'surprised', 'disgusted', 'contempt', 'neutral'):
            raise ValueError(f'Unknown emotion {label!r} for {name}; no neutral substitution')
        prior = ROOT / 'DICE-Talk-main/examples/emo' / (label + '.npy')
        if not prior.is_file():
            raise FileNotFoundError(prior)
        role = {'listener': 'supporter', 'speaker': 'seeker'}.get(entry.get('role'), entry.get('role'))
        cid = entry.get('post_id', entry.get('conversation', entry.get('dialogue_id')))
        identity = reference.rsplit('_', 1)[-1]
        items.append(dict(name=name, audio=str(audio_dir / name), portrait=str(image_dir / face),
                          prior=str(prior), emotion=label, identity=identity,
                          dialogue_id=cid, role=role, original_reference=entry))
    summary = dict(mapped_records=len(records), selected=len(items),
                   missing_audio=missing['missing_audio'], missing_portrait=missing['missing_portrait'],
                   unmapped_wavs=len(audio_names - records.keys()),
                   **audit,
                   emotions=dict(Counter(x['emotion'] for x in items)))
    if args.limit:
        items = items[:args.limit]
    summary['requested'] = len(items)
    return items, summary


def code_fingerprint():
    paths = [Path(__file__), ROOT / 'psymm_generation/core.py', ROOT / 'psymm_generation/long_video.py',
             ROOT / 'DICE-Talk-main/dice_talk.py', ROOT / 'DICE-Talk-main/src/dataset/test_preprocess.py',
             ROOT / 'DICE-Talk-main/config/inference/dice_talk.yaml']
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


@contextlib.contextmanager
def job_lock(path):
    import fcntl
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Video generation already running: {path}') from None
        yield


def database(path):
    db = sqlite3.connect(str(path), timeout=60)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS video (name TEXT PRIMARY KEY, signature TEXT, '
               'status TEXT, output_hash TEXT, details TEXT, updated REAL)')
    return db


def render(model, item, output, cache, options, seed):
    import soundfile as sf
    aq = sf.info(item['audio'])
    if aq.frames <= 0 or aq.samplerate <= 0:
        raise ValueError('Empty or invalid source WAV')
    duration = aq.frames / aq.samplerate

    portrait = cache / 'portraits' / (file_hash(item['portrait']) + '.png')
    portrait.parent.mkdir(parents=True, exist_ok=True)
    if not portrait.exists():
        check = model.preprocess(item['portrait'], expand_ratio=.35)
        if check['face_num'] != 1:
            raise ValueError(f"Expected one face, found {check['face_num']}")
        import cv2
        temporary = portrait.with_name(portrait.stem + f'.{os.getpid()}.png')
        try:
            model.crop_image(item['portrait'], str(temporary), check['crop_bbox'])
            image = cv2.imread(str(temporary))
            if image is None or min(image.shape[:2]) < 128:
                raise ValueError('Reference crop is missing or smaller than 128 pixels')
            os.replace(temporary, portrait)
        finally:
            temporary.unlink(missing_ok=True)
    if duration > options['video_segment_seconds']:
        from psymm_generation.long_video import segmented_video
        return segmented_video(model, Path(item['audio']), portrait, Path(item['prior']), output,
                               cache / 'parts' / output.stem, options, seed)
    temporary = output.with_name(output.stem + '.partial.mp4')
    try:
        rc = model.process(str(portrait), item['audio'], item['prior'], str(temporary),
                           min_resolution=512, inference_steps=options['steps'],
                           ref_scale=options['ref_scale'], emo_scale=options['emo_scale'], seed=seed)
        if rc != 0:
            raise RuntimeError(f'DICE generation failed: {rc}')
        qc = probe_video(temporary, duration, decode=True)
        os.replace(temporary, output)
        return dict(qc=qc, segmented=False)
    finally:
        temporary.unlink(missing_ok=True)
        temporary.with_name(temporary.stem + '_noaudio.mp4').unlink(missing_ok=True)


def run_items(index, items, settings):
    cache = Path(settings['cache'])
    output_dir = Path(settings['output'])
    options = settings['options']
    db = database(cache / 'state.sqlite')
    counts = dict(generated=0, resumed=0, failed=0, requested=len(items))
    model = None
    sys.path.insert(0, str(ROOT / 'DICE-Talk-main'))
    print(f'worker={index} GPU={settings["gpu"]} items={len(items)}', flush=True)
    for position, item in enumerate(items, 1):
        name = item['name']
        output = output_dir / (Path(name).stem + '.mp4')
        try:
            audio_hash = file_hash(item['audio'])
            image_hash = file_hash(item['portrait'])
            signature = digest([item, audio_hash, image_hash, file_hash(item['prior']), options, settings['seed']])
            saved = db.execute('SELECT signature,status,output_hash FROM video WHERE name=?', (name,)).fetchone()
            if saved and saved[0] == signature and saved[1] == 'complete' and output.exists():
                if file_hash(output) == saved[2]:
                    counts['resumed'] += 1
                    print(f'RESUME [{position}/{len(items)}] {name}', flush=True)
                    continue
            if model is None:
                from dice_talk import DICE_Talk
                model = DICE_Talk(device_id=0, enable_interpolate_frame=True)
            seed = int(digest([settings['seed'], name])[:8], 16) % (2**31)
            details = render(model, item, output, cache, options, seed)
            details.update(item, generated_video=str(output), source_audio_sha256=audio_hash,
                           reference_image_sha256=image_hash, seed=seed, signature=signature,
                           options=options, audio_provenance='existing legacy WAV; original synthesis not rerun',
                           dialogue_identity_policy='two distinct fixed identities per whole dialogue; portrait follows WAV reference')
            db.execute('INSERT OR REPLACE INTO video VALUES (?,?,?,?,?,?)',
                       (name, signature, 'complete', file_hash(output), json.dumps(details, ensure_ascii=False), time.time()))
            db.commit()
            with (cache / f'reference_map_{index:02d}.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(details, ensure_ascii=False) + '\n')
            counts['generated'] += 1
            print(f'COMPLETE [{position}/{len(items)}] {name}', flush=True)
        except Exception as error:
            db.rollback()
            db.execute('INSERT OR REPLACE INTO video VALUES (?,?,?,?,?,?)',
                       (name, '', 'failed', '', json.dumps(dict(error=str(error))), time.time()))
            db.commit()
            counts['failed'] += 1
            traceback.print_exc()
            if model is None:
                print('Model initialization failed; worker stopped. Fix the error and repeat to resume.', flush=True)
                break
        atomic_json(cache / f'worker_{index:02d}.summary.json', dict(counts, updated=time.time()))
    db.close()
    atomic_json(cache / f'worker_{index:02d}.summary.json', dict(counts, updated=time.time(), finished=True))
    return 1 if counts['failed'] else 0


def worker(index, items, settings):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(settings['gpu'])
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    with (Path(settings['cache']) / f'worker_{index:02d}.log').open('a', buffering=1, encoding='utf-8') as log:

        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        sys.stdout = sys.stderr = log
        try:
            raise SystemExit(run_items(index, items, settings))
        except Exception:
            traceback.print_exc()
            raise


def main(default_dataset=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, default=default_dataset or ROOT / 'infer_outputs_extes')
    parser.add_argument('--audio-map', type=Path)
    parser.add_argument('--audio-dir', type=Path)
    parser.add_argument('--image-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--instances', '--workers-per-gpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=25)
    parser.add_argument('--ref-scale', '--ref_scale', type=float, default=3.)
    parser.add_argument('--emo-scale', '--emo_scale', type=float, default=6.)
    parser.add_argument('--video-segment-seconds', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true', help='Read map and directory names; no writes or CUDA imports')
    args = parser.parse_args()
    if args.instances < 1 or args.gpu < 0 or args.steps < 1 or not 2 <= args.video_segment_seconds <= 60 or args.limit < 0:
        parser.error('GPU/limit must be nonnegative; instances/steps positive; segment seconds 2..60')
    items, summary = plan(args)
    print(json.dumps(dict(summary, gpu=args.gpu, instances=args.instances), ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print('DRY RUN: no GPU, media contents, output writes, or TTS.', flush=True)
        return 0
    require_fixed_identities(summary)
    if not items:
        raise ValueError('No mapped WAVs with paired portraits')

    os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
    os.environ.setdefault('TORCH_HOME', str(ROOT / '.cache/torch'))
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    output_dir = (args.output_dir or args.dataset_dir / 'DICE_Talk_outputs').resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir.parent / ('.' + output_dir.name + '_generation')
    cache.mkdir(exist_ok=True)
    options = dict(steps=args.steps, ref_scale=args.ref_scale, emo_scale=args.emo_scale,
                   video_segment_seconds=args.video_segment_seconds,
                   _configuration_hash=digest(code_fingerprint()))
    settings = dict(gpu=args.gpu, cache=str(cache), output=str(output_dir), options=options, seed=args.seed)
    with job_lock(cache / 'job.lock'):

        database(cache / 'state.sqlite').close()
        atomic_json(cache / 'launch.json', dict(summary, settings=settings, instances=args.instances, started=time.time()))
        context = mp.get_context('spawn')
        processes = []
        def stop(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop)
        try:
            for index in range(args.instances):
                shard = items[index::args.instances]
                if not shard:
                    continue
                p = context.Process(target=worker, args=(index, shard, settings))
                p.start()
                processes.append(p)
            while any(p.is_alive() for p in processes):
                time.sleep(10)
                counts = dict(database_counts(cache / 'state.sqlite'))
                print(f'{time.strftime("%H:%M:%S")} {counts} / {len(items)}; logs: {cache}', flush=True)
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            for p in processes:
                if p.is_alive():
                    p.terminate()
            for p in processes:
                p.join()
            print('Interrupted. Repeat the same command to resume.', flush=True)
            return 130
        codes = [p.exitcode for p in processes]
        atomic_json(cache / 'summary.json', dict(exit_codes=codes, counts=dict(database_counts(cache / 'state.sqlite'))))
        print(f'Finished: exit_codes={codes}; repeat command to retry failures.', flush=True)
        return 1 if any(codes) or summary['missing_audio'] or summary['missing_portrait'] else 0


def database_counts(path):
    with sqlite3.connect(str(path), timeout=60) as db:
        return db.execute('SELECT status,count(*) FROM video GROUP BY status').fetchall()


if __name__ == '__main__':
    raise SystemExit(main())
