import argparse
import contextlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from .core import (ROOT, WEIGHTS, DATA, INPUTS, atomic_json, digest, file_hash,
                   normalized_dialogue, probe_audio, probe_video)
from .references import verify_reference


@contextlib.contextmanager
def run_lock(run, stage=None):
    import fcntl
    run.mkdir(parents=True, exist_ok=True)

    lock_name = '.generation.video.lock' if stage == 'video' else '.generation.lock'
    with (run / lock_name).open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def read_plan(run):
    meta = json.loads((run / 'plan.meta.json').read_text(encoding='utf-8'))
    if meta.get('schema_version') != 2:
        raise ValueError('This plan predates dialogue emotion-pool anchors; prepare a new --run')
    if file_hash(run / 'plan.jsonl') != meta['plan_sha256']:
        raise ValueError('Plan changed; prepare a new run directory')
    with (run / 'plan.jsonl').open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare(args):
    from .references import (catalog_from_metadata, emotion_pool_from_filenames,
                             select_dialogue_references)
    run = args.run.resolve()
    with run_lock(run):
        if (run / 'plan.meta.json').exists() or (run / 'state.sqlite').exists():
            raise ValueError('Run already prepared; resume audio/video or choose a new --run')
        source_path = Path(args.input).resolve() if args.input else ROOT / INPUTS[args.source]
        data = json.loads(source_path.read_text(encoding='utf-8'))
        if not isinstance(data, list):
            raise ValueError('Input JSON must be a list of dialogue records')
        catalog_path = args.catalog or DATA / 'AvaMERG/train.json'
        catalog_records = json.loads(Path(catalog_path).read_text(encoding='utf-8'))
        catalog = catalog_from_metadata(catalog_records, DATA / 'AvaMERG')
        pool_audit = {}
        pool = emotion_pool_from_filenames(catalog, DATA / 'AvaMERG/emo', audit=pool_audit)
        atomic_json(run / 'emotion_pool_index.json', pool_audit)
        count, dialogue_count, rejected = 0, 0, []
        selected = data[args.offset:args.offset + args.limit] if args.limit else data[args.offset:]
        tmp = run / 'plan.jsonl.tmp'
        with tmp.open('w', encoding='utf-8') as out:
            for index, record in enumerate(selected, args.offset):
                try:
                    dialogues = list(normalized_dialogue(args.source, record, index, args.qa_answers))
                except (ValueError, KeyError, TypeError) as error:
                    rejected.append(dict(source_index=index, record_sha256=digest(record),
                                         error=str(error)))
                    if args.invalid_record_policy == 'fail':
                        atomic_json(run / 'rejected_records.json', rejected)
                        raise
                    continue
                for dialogue in dialogues:
                    references = select_dialogue_references(catalog, pool, dialogue,
                                                            run / 'references', args.seed)
                    turns = dialogue.pop('turns')
                    if args.turn_limit:
                        turns = turns[:args.turn_limit]
                    for turn in turns:
                        asset_id = f"{dialogue['dialogue_id']}_T{turn['turn_index']:04d}"
                        item = dict(dialogue, **turn, asset_id=asset_id,
                                    seed=int(digest([args.seed, asset_id])[:8], 16) % (2**31),
                                    reference=references[turn['role']])
                        out.write(json.dumps(item, ensure_ascii=False) + '\n')
                        count += 1
                    dialogue_count += 1
        if not count:
            raise ValueError('No turns selected')
        os.replace(tmp, run / 'plan.jsonl')
        atomic_json(run / 'rejected_records.json', rejected)
        atomic_json(run / 'plan.meta.json', dict(schema_version=2, source=str(source_path),
                    source_sha256=file_hash(source_path), catalog=str(catalog_path),
                    catalog_sha256=file_hash(catalog_path), plan_sha256=file_hash(run / 'plan.jsonl'),
                    seed=args.seed, dialogues=dialogue_count, turns=count,
                    offset=args.offset, limit=args.limit, turn_limit=args.turn_limit,
                    invalid_record_policy=args.invalid_record_policy, rejected_records=len(rejected),
                    qa_answer_policy=args.qa_answers, created_at=time.time(),
                    emotion_pool_membership_sha256=digest(pool),
                    policy='Two distinct identities per dialogue; fixed paired audio/portrait per role. '
                    'Non-neutral audio references come from the matching AvaMERG emotion pool; '
                    'neutral uses base TTS expression with identity conversion, no neutral pool. '
                    'Turn annotation > seeker dialogue emotion > original source supporter policy '
                    '> unannotated neutral. Emotion transfer requires perceptual validation.'))
        print(json.dumps(dict(run=str(run), dialogues=dialogue_count, turns=count,
                              rejected_records=len(rejected)), ensure_ascii=False))


def inventory():

    paths = [* (ROOT / 'psymm_generation').glob('*.py'),
             ROOT / 'OpenVoice-main/openvoice/api.py', ROOT / 'OpenVoice-main/openvoice/se_extractor.py',
             ROOT / 'DICE-Talk-main/dice_talk.py', ROOT / 'DICE-Talk-main/src/dataset/test_preprocess.py',
             ROOT / 'DICE-Talk-main/config/inference/dice_talk.yaml',
             ROOT / 'MeloTTS-main/melo/api.py', ROOT / 'MeloTTS-main/melo/utils.py',
              * (ROOT / 'MeloTTS-main/melo/text').glob('*.py'),
              * (ROOT / 'DICE-Talk-main/examples/emo').glob('*.npy')]
    code = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    models = [WEIGHTS / 'openvoice_weights/converter/checkpoint.pth',
              WEIGHTS / 'openvoice_weights/converter/config.json',
              WEIGHTS / 'openvoice_weights/base_speakers/ses/en-newest.pth',
              WEIGHTS / 'openvoice_weights/base_speakers/ses/zh.pth',
              WEIGHTS / 'dicetalk_weights/DICE-Talk/unet.pth',
              WEIGHTS / 'dicetalk_weights/DICE-Talk/audio_linear.pth',
              WEIGHTS / 'dicetalk_weights/DICE-Talk/pose_guider.pth',
              WEIGHTS / 'dicetalk_weights/DICE-Talk/emo_model.pth',
              WEIGHTS / 'dicetalk_weights/yoloface_v5m.pt']
    for name in ('MeloTTS-Chinese', 'MeloTTS-English-v3'):
        models += [ROOT / 'MeloTTS-main/myshell-ai' / name / f for f in ('config.json', 'checkpoint.pth')]


    for directory in (WEIGHTS / 'bert-base-uncased', WEIGHTS / 'bert-base-multilingual-uncased',
                      WEIGHTS / 'dicetalk_weights/whisper-tiny',
                      WEIGHTS / 'dicetalk_weights/stable-video-diffusion-img2vid-xt',
                      WEIGHTS / 'dicetalk_weights/RIFE'):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        models += [p for p in directory.rglob('*') if p.is_file() and
                   p.suffix in ('.json', '.pth', '.pt', '.pkl', '.bin', '.safetensors', '.txt')]
    models = sorted(set(models))
    weights = {str(p): dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns,
                           sha256=file_hash(p) if p.stat().st_size < 1024*1024 else None) for p in models}
    git = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return dict(code=code, weights=weights, git_commit=git.stdout.strip(),
                python=sys.version, executable=sys.executable,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))


def database(run):
    db = sqlite3.connect(str(run / 'state.sqlite'))
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS media (asset TEXT, stage TEXT, signature TEXT, '
               'status TEXT, path TEXT, sha256 TEXT, details TEXT, updated REAL, PRIMARY KEY(asset, stage))')
    return db


def reusable(row, signature, output, stage, audio=None):
    if not row or row[0] != signature or row[1] != 'complete' or not output.exists():
        return False
    try:
        if len(row) > 3:
            details = json.loads(row[3])
            if details.get('base_audio') and file_hash(details['base_audio']) != details['base_audio_sha256']:
                return False
        if file_hash(output) != row[2]:
            return False
        if stage == 'audio':
            probe_audio(output)
        else:
            probe_video(output, probe_audio(audio)['seconds'], decode=True)
        return True
    except Exception:
        return False


def verified_record(db, run, item, stage, info):
    row = db.execute('SELECT path,sha256,status,details,signature FROM media WHERE asset=? AND stage=?',
                     (item['asset_id'], stage)).fetchone()
    if not row or row[2] != 'complete' or file_hash(row[0]) != row[1]:
        raise ValueError(f'Missing, failed or modified {stage}: {item["asset_id"]}')
    details = json.loads(row[3])
    config_hash = details['configuration_hash']
    config = json.loads((run / 'configurations' / (config_hash + '.json')).read_text())
    signature_config = dict(code=config['inventory']['code'], weights=config['inventory']['weights'],
                            options=config['options'], stage=config['stage'])
    if digest(signature_config) != config_hash or config['stage'] != stage:
        raise ValueError('Changed configuration snapshot')
    if config['inventory']['code'] != info['code'] or config['inventory']['weights'] != info['weights']:
        raise ValueError(f'Stale {stage} code/model fingerprint; rerun {stage}')
    dependency = verified_record(db, run, item, 'audio', info)[1] if stage == 'video' else None
    if digest([item, config_hash, dependency]) != row[4]:
        raise ValueError(f'Stale {stage} input or dependency; rerun {stage}')
    verify_reference(item)
    if details.get('base_audio') and file_hash(details['base_audio']) != details['base_audio_sha256']:
        raise ValueError('Base TTS audio changed')
    return row


def generate(args):
    run = args.run.resolve()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or ',' in os.environ['CUDA_VISIBLE_DEVICES']:
        raise ValueError('Select exactly one GPU with --gpu or CUDA_VISIBLE_DEVICES')
    os.environ['PSYMM_WEIGHTS'] = str(WEIGHTS)
    with run_lock(run):
        items = read_plan(run)
        if args.limit:
            items = items[:args.limit]
        info = inventory()
        options = dict(speed=args.speed, sdp_ratio=args.sdp_ratio, noise_scale=args.noise_scale,
                       noise_scale_w=args.noise_scale_w, tau=args.tau, reference_method=args.reference_method,
                       steps=args.steps, ref_scale=args.ref_scale, emo_scale=args.emo_scale,
                       max_video_seconds=args.max_video_seconds,
                       video_segment_seconds=args.video_segment_seconds)
        config = dict(inventory=info, options=options, stage=args.command)

        config_hash = digest(dict(code=info['code'], weights=info['weights'], options=options,
                                  stage=args.command))
        atomic_json(run / 'configurations' / (config_hash + '.json'), dict(config, configuration_hash=config_hash))
        atomic_json(run / (args.command + '.config.json'), dict(config, configuration_hash=config_hash))
        from .workers import AudioWorker, VideoWorker
        worker = None
        db, failed, complete, skipped = database(run), 0, 0, 0
        stage = args.command
        folder = run / stage
        folder.mkdir(exist_ok=True)
        for item in items:
            output = folder / (item['asset_id'] + ('.wav' if stage == 'audio' else '.mp4'))
            audio = run / 'audio' / (item['asset_id'] + '.wav')
            try:
                verify_reference(item)
                dependency = None
                if stage == 'video':
                    saved = verified_record(db, run, item, 'audio', info)
                    dependency = saved[1]
                signature = digest([item, config_hash, dependency])
                row = db.execute('SELECT signature,status,sha256,details FROM media WHERE asset=? AND stage=?',
                                 (item['asset_id'], stage)).fetchone()
                if reusable(row, signature, output, stage, audio):
                    skipped += 1
                    print(f"RESUME {stage} {item['asset_id']}", flush=True)
                    continue
                if worker is None:
                    worker = (AudioWorker if stage == 'audio' else VideoWorker)(
                        run, dict(options, _configuration_hash=config_hash))
                details = worker.generate(item, output)
                details.update(configuration_hash=config_hash, input_signature=signature)
                db.execute('INSERT OR REPLACE INTO media VALUES (?,?,?,?,?,?,?,?)',
                           (item['asset_id'], stage, signature, 'complete', str(output),
                            file_hash(output), json.dumps(details, ensure_ascii=False), time.time()))
                db.commit()
                complete += 1
                print(f"COMPLETE {stage} {item['asset_id']} {details['qc']}", flush=True)
            except Exception as error:
                import traceback
                traceback.print_exc()
                db.execute('INSERT OR REPLACE INTO media VALUES (?,?,?,?,?,?,?,?)',
                           (item['asset_id'], stage, '', 'failed', str(output), '',
                            json.dumps(dict(error=str(error))), time.time()))
                db.commit()
                failed += 1
                if not args.keep_going:
                    break
        db.close()
        summary = dict(stage=stage, generated=complete, resumed=skipped, failed=failed, requested=len(items))
        atomic_json(run / (stage + '.summary.json'), summary)
        print(json.dumps(summary), flush=True)
        return 1 if failed else 0


def validate(args):
    run = args.run.resolve()
    with run_lock(run):
        db = database(run)
        info = inventory()
        report, failures = [], 0
        for item in read_plan(run):
            record = dict(asset_id=item['asset_id'], role=item['role'], language=item['language'],
                          intended_emotion=item['emotion'], emotion_origin=item['emotion_origin'])
            for stage in ('audio', 'video'):
                try:
                    row = verified_record(db, run, item, stage, info)
                    aq = probe_audio(run / 'audio' / (item['asset_id'] + '.wav'))
                    record[stage] = aq if stage == 'audio' else probe_video(row[0], aq['seconds'], True)
                except Exception as e:
                    record[stage] = dict(error=str(e))
                    failures += 1
            report.append(record)
        atomic_json(run / 'validation.json', dict(records=report, failed_checks=failures,
                    semantic_emotion_and_lipsync='pending independent perceptual evaluation'))
        print(json.dumps(dict(turns=len(report), failed_checks=failures)))
        return 1 if failures else 0


def export(args):
    run = args.run.resolve()
    with run_lock(run):
        items = read_plan(run)
        db = database(run)
        info = inventory()
        records = []
        for item in items:
            media = {}
            for stage in ('audio', 'video'):
                row = verified_record(db, run, item, stage, info)
                aq = probe_audio(run / 'audio' / (item['asset_id'] + '.wav'))
                if stage == 'video':
                    probe_video(row[0], aq['seconds'], True)
                media[stage] = dict(path=row[0], sha256=row[1], generation=json.loads(row[3]))
            records.append(dict(item, media=media, perceptual_review_status='pending'))
        atomic_json(run / 'multimodal_manifest.json', records)
        print(f'Exported {len(records)} verified media mappings; human/semantic review remains pending')


def preflight(args):
    import shutil
    info = inventory()
    for command in ('ffmpeg', 'ffprobe'):
        if not shutil.which(command):
            raise FileNotFoundError(command)
    for label in ('neutral', 'fear', 'sad', 'angry', 'happy', 'surprised', 'disgusted', 'contempt'):
        if not (ROOT / 'DICE-Talk-main/examples/emo' / (label + '.npy')).exists():
            raise FileNotFoundError(label)
    print(json.dumps(dict(root=str(ROOT), weights=str(WEIGHTS), data=str(DATA),
                         checked_models=len(info['weights']), git=info['git_commit']), indent=2))


def direct_database(run):
    db = sqlite3.connect(str(run / 'direct_state.sqlite'), timeout=120)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS items (asset_id TEXT PRIMARY KEY, source_index INTEGER, '
               'dialogue_id TEXT, item TEXT NOT NULL)')
    db.execute('CREATE INDEX IF NOT EXISTS items_source ON items(source_index)')
    db.execute('CREATE INDEX IF NOT EXISTS items_dialogue ON items(dialogue_id)')
    db.execute('CREATE TABLE IF NOT EXISTS media (asset_id TEXT, stage TEXT, signature TEXT, '
               'status TEXT, sha256 TEXT, details TEXT, PRIMARY KEY(asset_id, stage))')
    db.execute('CREATE TABLE IF NOT EXISTS skipped (source_index INTEGER PRIMARY KEY, reason TEXT)')
    db.commit()
    return db


def repair_short_references(db, items, catalog, pool, cache, seed, failed_assets):
    from .references import select_anchor
    roles = {item['role'] for item in items if item['asset_id'] in failed_assets}
    for role in sorted(roles):
        members = [item for item in items if item['role'] == role]
        original = members[0]['reference']

        if (original.get('replaces_reference')
                and original.get('audio_qc', {}).get('vad_speech_seconds', 0) >= 1.5
                and all(item['reference'] == original for item in members)):
            continue
        if any(db.execute('SELECT 1 FROM media WHERE asset_id=? AND status=?',
                          (item['asset_id'], 'complete')).fetchone() for item in members):
            raise ValueError('Cannot replace a reference for a role with completed media')
        label = members[0]['emotion']
        choices = catalog if label == 'neutral' else pool[label]
        candidates = [reference for reference in choices[original['identity']]
                      if reference['segment'] != original['segment']]
        replacement = select_anchor({original['identity']: candidates}, original['identity'],
                                    cache, seed,
                                    scope=f"{members[0]['dialogue_id']}:{role}:vad-repair:{original['audio_sha256']}",
                                    target_emotion=label, validate_speech=True)
        replacement = dict(replacement, replaces_reference=dict(
            segment=original['segment'], audio=original['audio'],
            audio_sha256=original['audio_sha256'], portrait=original['portrait'],
            reason='VAD found less than 1.5 seconds of speech'))
        for item in members:
            item['reference'] = replacement
        with db:
            db.executemany('UPDATE items SET item=? WHERE asset_id=?',
                           [(json.dumps(item, ensure_ascii=False), item['asset_id'])
                            for item in members])
        print(f"REFERENCE REPAIRED {members[0]['dialogue_id']} {role} "
              f"identity={original['identity']} emotion={label}: "
              f"{original['segment']} -> {replacement['segment']}", flush=True)


def direct_audio_worker(index, total, gpu, source, run, seed, options):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
    os.environ.setdefault('TORCH_HOME', str(ROOT / '.cache/torch'))
    os.environ.setdefault('NLTK_DATA', str(ROOT / '.cache/nltk'))
    from .references import (catalog_from_metadata, emotion_pool_from_filenames,
                             select_dialogue_references)
    from .workers import AudioWorker
    db = direct_database(run)
    data = json.loads((ROOT / INPUTS[source]).read_text(encoding='utf-8'))
    catalog_records = json.loads((DATA / 'AvaMERG/train.json').read_text(encoding='utf-8'))
    catalog = catalog_from_metadata(catalog_records, DATA / 'AvaMERG')
    pool = emotion_pool_from_filenames(catalog, DATA / 'AvaMERG/emo')
    worker = None
    counts = dict(generated=0, resumed=0, failed=0, skipped=0)
    failed_assets = {asset_id for asset_id, details in db.execute(
        'SELECT asset_id,details FROM media WHERE stage=? AND status=?', ('audio', 'failed'))
        if 'VAD found less than 1.5 seconds of speech' in details}
    for source_index, record in enumerate(data):
        if source_index % total != index:
            continue
        try:
            if source == 'extes':
                record = dict(record)
                record['dialog'] = [turn for turn in record.get('dialog', [])
                                    if isinstance(turn.get('content'), str) and turn['content'].strip()]
                roles = {turn.get('speaker') for turn in record['dialog']}
                if not {'seeker', 'supporter'} <= roles:
                    db.execute('INSERT OR REPLACE INTO skipped VALUES (?,?)',
                               (source_index, 'No nonblank seeker/supporter pair'))
                    db.commit()
                    counts['skipped'] += 1
                    continue
            for dialogue in normalized_dialogue(source, record, source_index, qa_answers='all'):
                rows = db.execute('SELECT item FROM items WHERE dialogue_id=? ORDER BY asset_id',
                                  (dialogue['dialogue_id'],)).fetchall()
                if rows:
                    items = [json.loads(row[0]) for row in rows]
                    if len(items) != len(dialogue['turns']):
                        raise ValueError('Saved dialogue has a different number of turns')
                    if any(item['asset_id'] in failed_assets for item in items):
                        repair_short_references(db, items, catalog, pool,
                                                run / 'references' / f'worker_{index}',
                                                seed, failed_assets)
                else:
                    refs = select_dialogue_references(catalog, pool, dialogue,
                                                       run / 'references' / f'worker_{index}', seed,
                                                       validate_speech=True)
                    items = []
                    for turn in dialogue['turns']:
                        asset_id = f"{dialogue['dialogue_id']}_T{turn['turn_index']:04d}"
                        item = dict((k, v) for k, v in dialogue.items() if k != 'turns')
                        item.update(turn, asset_id=asset_id,
                                    seed=int(digest([seed, asset_id])[:8], 16) % (2**31),
                                    reference=refs[turn['role']])
                        items.append(item)
                    db.executemany('INSERT INTO items VALUES (?,?,?,?)',
                                   [(item['asset_id'], source_index, dialogue['dialogue_id'],
                                     json.dumps(item, ensure_ascii=False)) for item in items])
                    db.commit()
                for item in items:
                    output = run / 'audio' / (item['asset_id'] + '.wav')
                    signature = digest([item, options])
                    saved = db.execute('SELECT signature,status,sha256 FROM media '
                                       'WHERE asset_id=? AND stage=?',
                                       (item['asset_id'], 'audio')).fetchone()
                    if saved and saved[0] == signature and saved[1] == 'complete' and output.exists() \
                            and file_hash(output) == saved[2]:
                        counts['resumed'] += 1
                        continue
                    try:
                        if worker is None:
                            worker = AudioWorker(run, options)
                        details = worker.generate(item, output)
                        db.execute('INSERT OR REPLACE INTO media VALUES (?,?,?,?,?,?)',
                                   (item['asset_id'], 'audio', signature, 'complete',
                                    file_hash(output), json.dumps(details, ensure_ascii=False)))
                        db.commit()
                        counts['generated'] += 1
                        print(f"AUDIO {index} {item['asset_id']} {details['qc']['seconds']:.1f}s", flush=True)
                    except Exception as error:
                        import traceback
                        traceback.print_exc()
                        db.execute('INSERT OR REPLACE INTO media VALUES (?,?,?,?,?,?)',
                                   (item['asset_id'], 'audio', signature, 'failed', '',
                                    json.dumps({'error': str(error)})))
                        db.commit()
                        counts['failed'] += 1
                        if worker is None:
                            db.close()
                            return 1
        except Exception:
            import traceback
            traceback.print_exc()
            counts['failed'] += 1
            continue
    db.close()
    print(f'AUDIO WORKER {index} {counts}', flush=True)
    return 1 if counts['failed'] else 0


def direct_video_worker(index, total, gpu, run, options):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
    os.environ.setdefault('TORCH_HOME', str(ROOT / '.cache/torch'))
    from .workers import VideoWorker
    db = direct_database(run)
    rows = db.execute('SELECT item FROM items WHERE source_index % ? = ? '
                      'ORDER BY source_index, asset_id', (total, index)).fetchall()
    worker = None
    counts = dict(generated=0, resumed=0, failed=0, missing_audio=0)
    for (record,) in rows:
        item = json.loads(record)
        asset_id = item['asset_id']
        audio = run / 'audio' / (asset_id + '.wav')
        audio_row = db.execute('SELECT status,sha256 FROM media WHERE asset_id=? AND stage=?',
                               (asset_id, 'audio')).fetchone()
        if not audio_row or audio_row[0] != 'complete' or not audio.exists():
            counts['missing_audio'] += 1
            continue
        if file_hash(audio) != audio_row[1]:
            print(f'CHANGED AUDIO {asset_id}; rerun infer_voice_emo.py', flush=True)
            counts['missing_audio'] += 1
            continue
        output = run / 'video' / (asset_id + '.mp4')
        signature = digest([item, audio_row[1], options])
        saved = db.execute('SELECT signature,status,sha256 FROM media '
                           'WHERE asset_id=? AND stage=?', (asset_id, 'video')).fetchone()
        if saved and saved[0] == signature and saved[1] == 'complete' and output.exists() \
                and file_hash(output) == saved[2]:
            counts['resumed'] += 1
            continue
        try:
            if worker is None:
                worker = VideoWorker(run, options)
            details = worker.generate(item, output)
            db.execute('INSERT OR REPLACE INTO media VALUES (?,?,?,?,?,?)',
                       (asset_id, 'video', signature, 'complete', file_hash(output),
                        json.dumps(details, ensure_ascii=False)))
            db.commit()
            counts['generated'] += 1
            print(f"VIDEO {index} {asset_id} {details['qc']['video_seconds']:.1f}s", flush=True)
        except Exception as error:
            import traceback
            traceback.print_exc()
            db.execute('INSERT OR REPLACE INTO media VALUES (?,?,?,?,?,?)',
                       (asset_id, 'video', signature, 'failed', '',
                        json.dumps({'error': str(error)})))
            db.commit()
            counts['failed'] += 1
            if worker is None:
                break
    db.close()
    print(f'VIDEO WORKER {index} {counts}', flush=True)
    return 1 if counts['failed'] or counts['missing_audio'] else 0


def _direct_entry_unlocked(stage, source, gpu_ids, workers_per_gpu, run, seed):
    """Run one dataset directly, selecting references only as each dialogue starts."""
    import multiprocessing as mp
    if stage not in ('audio', 'video') or source not in ('extes', 'psydial', 'psyqa'):
        raise ValueError('Unknown direct generation stage or dataset')
    if not gpu_ids or any(not isinstance(gpu, int) or gpu < 0 for gpu in gpu_ids):
        raise ValueError('GPU_IDS must contain nonnegative GPU numbers')
    if workers_per_gpu < 1:
        raise ValueError('WORKERS_PER_GPU must be at least one')
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    (run / stage).mkdir(exist_ok=True)
    (run / 'references').mkdir(exist_ok=True)
    db = direct_database(run)
    expected = dict(source=source, seed=seed,
                    input_sha256=file_hash(ROOT / INPUTS[source]))
    stored = db.execute('SELECT value FROM metadata WHERE key=?', ('source',)).fetchone()
    if stored and json.loads(stored[0]) != expected:
        raise ValueError('Input dataset or reference seed changed; use a new output directory')
    db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',
               ('source', json.dumps(expected)))
    if stage == 'video' and not db.execute('SELECT 1 FROM items LIMIT 1').fetchone():
        db.close()
        raise ValueError('No generated audio plan found; run infer_voice_emo.py first')
    db.commit()
    db.close()
    options = dict(speed=.9, sdp_ratio=.2, noise_scale=.3, noise_scale_w=.6,
                   tau=.3, reference_method='silero_vad', steps=25,
                   ref_scale=3., emo_scale=6., max_video_seconds=120.,
                   video_segment_seconds=20)
    options['_configuration_hash'] = digest(options)
    targets = [(gpu, slot) for gpu in gpu_ids for slot in range(workers_per_gpu)]
    context = mp.get_context('spawn')
    processes = []
    for index, (gpu, _) in enumerate(targets):
        target = direct_audio_worker if stage == 'audio' else direct_video_worker
        args = ((index, len(targets), gpu, source, run, seed, options) if stage == 'audio'
                else (index, len(targets), gpu, run, options))
        process = context.Process(target=target, args=args)
        process.start()
        processes.append(process)
    print(f'{source} {stage}: GPU_IDS={gpu_ids}, WORKERS_PER_GPU={workers_per_gpu}, '
          f'output={run}', flush=True)
    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
        for process in processes:
            process.join()
        raise
    if stage == 'audio':
        db = direct_database(run)
        mapping = {}
        for asset_id, record in db.execute('SELECT items.asset_id,items.item FROM items '
                                           'JOIN media ON items.asset_id=media.asset_id '
                                           'WHERE media.stage=? AND media.status=?',
                                           ('audio', 'complete')):
            item = json.loads(record)
            ref = item['reference']
            mapping[asset_id + '.wav'] = dict(dialogue_id=item['dialogue_id'],
                source_id=item['source_id'], answer_index=item['answer_index'],
                turn=item['turn_index'], role=item['role'], emotion=item['emotion'],
                identity=ref['identity'], reference_audio=Path(ref['audio']).name,
                reference_path=ref['audio'], reference_image=ref['portrait'],
                generated_audio=str(run / 'audio' / (asset_id + '.wav')))
        atomic_json(run.parent / 'audio_reference_map_emo_direct.json', mapping)
        if source == 'extes':
            skipped = [dict(source_index=row[0], reason=row[1]) for row in
                       db.execute('SELECT source_index,reason FROM skipped ORDER BY source_index')]
            atomic_json(run / 'skipped_dialogues.json', skipped)
        db.close()
        print(f'Audio reference map: {run.parent / "audio_reference_map_emo_direct.json"} '
              f'({len(mapping)} entries)', flush=True)
    failures = [(p.pid, p.exitcode) for p in processes if p.exitcode]
    if failures:
        print(f'Workers with errors: {failures}', flush=True)
        return 1
    return 0


def direct_entry(stage, source, gpu_ids, workers_per_gpu, run, seed=20261015):
    with run_lock(Path(run).resolve(), stage=stage):
        return _direct_entry_unlocked(stage, source, gpu_ids, workers_per_gpu, run, seed)


def main():

    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    cache = ROOT / '.cache'
    os.environ.setdefault('HF_HOME', str(cache / 'huggingface'))
    os.environ.setdefault('TORCH_HOME', str(cache / 'torch'))
    os.environ.setdefault('NLTK_DATA', str(cache / 'nltk'))
    parser = argparse.ArgumentParser(description='PsyMM reproducible multimodal generation')
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('prepare')
    p.add_argument('--source', choices=INPUTS, required=True)
    p.add_argument('--input')
    p.add_argument('--catalog', type=Path)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--limit', type=int, default=0, help='Number of source records; zero selects all')
    p.add_argument('--turn-limit', type=int, default=0, help='Explicit smoke subset only')
    p.add_argument('--seed', type=int, default=20261015)
    p.add_argument('--qa-answers', choices=('first', 'all'), default='first')
    p.add_argument('--invalid-record-policy', choices=('fail', 'quarantine'), default='fail',
                   help='Default fails visibly; quarantine writes every rejected source index and reason')
    p.set_defaults(func=prepare)
    for stage in ('audio', 'video'):
        p = subs.add_parser(stage)
        p.add_argument('--run', type=Path, required=True)
        p.add_argument('--gpu', type=int)
        p.add_argument('--limit', type=int, default=0)
        p.add_argument('--keep-going', action='store_true')
        p.add_argument('--speed', type=float, default=.9)
        p.add_argument('--sdp-ratio', type=float, default=.2)
        p.add_argument('--noise-scale', type=float, default=.3)
        p.add_argument('--noise-scale-w', type=float, default=.6)
        p.add_argument('--tau', type=float, default=.3)
        p.add_argument('--reference-method', choices=('silero_vad', 'pretrimmed'), default='silero_vad')
        p.add_argument('--steps', type=int, default=25)
        p.add_argument('--ref-scale', type=float, default=3.)
        p.add_argument('--emo-scale', type=float, default=6.)
        p.add_argument('--max-video-seconds', type=float, default=120.)
        p.add_argument('--video-segment-seconds', type=int, default=0,
                       help='Opt-in bounded video generation; 0 keeps the original duration limit')
        p.set_defaults(func=generate)
    for name, function in (('validate', validate), ('export', export)):
        p = subs.add_parser(name)
        p.add_argument('--run', type=Path, required=True)
        p.set_defaults(func=function)
    p = subs.add_parser('preflight')
    p.set_defaults(func=preflight)
    args = parser.parse_args()
    if getattr(args, 'video_segment_seconds', 0) not in (0, *range(2, 61)):
        parser.error('video-segment-seconds must be 0 or an integer from 2 to 60')
    if getattr(args, 'speed', 1) <= 0 or getattr(args, 'steps', 25) < 1:
        parser.error('speed and steps must be positive')
    if not 0 <= getattr(args, 'sdp_ratio', .2) <= 1:
        parser.error('sdp-ratio must be between zero and one')
    if min(getattr(args, 'noise_scale', .3), getattr(args, 'noise_scale_w', .6),
           getattr(args, 'tau', .3)) < 0:
        parser.error('noise parameters must be nonnegative')
    return args.func(args) or 0
