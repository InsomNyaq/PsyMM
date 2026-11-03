#!/usr/bin/env python3
"""PsyQA tmux orchestration. launch starts GPUs; --dry-run/status never do."""
import argparse
import contextlib
import json
import os
import re
import shlex
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from psymm_generation.core import INPUTS, atomic_json, digest, file_hash
from psymm_generation.cli import read_plan
from psyqa_identity import validate_items


def partitions(count, workers):
    if not 1 <= workers <= count:
        raise ValueError('Instance count must be between 1 and question count')
    return [dict(index=i, offset=count*i//workers,
                 limit=count*(i+1)//workers-count*i//workers) for i in range(workers)]


def gpu_list(value, per_gpu):
    if not re.fullmatch(r'\d+(,\d+)*', value) or per_gpu < 1:
        raise ValueError('Use GPU IDs such as 0,2 and a positive instances-per-gpu')
    ids = value.split(',')
    if len(ids) != len(set(ids)):
        raise ValueError('GPU IDs must be distinct; use instances-per-gpu to repeat a GPU')
    return [int(g) for g in ids for _ in range(per_gpu)]


@contextlib.contextmanager
def lock(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def reference_maps(run):
    """Explicit new schema; never overwrite historical infer_outputs maps."""
    audio, images, identities, emotions = {}, {}, Counter(), Counter()
    role_keys = set()
    items = read_plan(run)

    validate_items(items)
    for item in items:
        ref = item['reference']
        base = {k: item[k] for k in ('source_index', 'source_id', 'dialogue_id',
                                     'answer_index', 'role', 'emotion', 'emotion_origin')}
        base.update(identity=ref['identity'], segment=ref['segment'],
                    reference_source=ref['reference_source'], reference_emotion=ref['reference_emotion'])
        audio[item['asset_id']] = dict(base, reference_audio=ref['audio'],
                                      sha256=ref['audio_sha256'],
                                      output_audio=str(run/'audio'/(item['asset_id']+'.wav')))
        images[item['asset_id']] = dict(base, reference_image=ref['portrait'],
                                       sha256=ref['portrait_sha256'], reference_video=ref['video'],
                                       output_video=str(run/'video'/(item['asset_id']+'.mp4')))
        role_key = (item['dialogue_id'], item['role'])
        if role_key not in role_keys:
            identities[ref['identity']] += 1
            role_keys.add(role_key)
        emotions[item['emotion']] += 1
    for name, mapping in [('audio_reference_map_emo.json', audio), ('ref_map.json', images)]:
        atomic_json(run/name, dict(schema_version=1, key='asset_id',
                                  note='Planned references, not proof of completed generation', records=mapping))
    atomic_json(run/'reference_usage.json', dict(planned_turns=len(items),
                distinct_identities=len(identities), dialogue_role_count_by_identity=dict(identities),
                turn_count_by_intended_emotion=dict(emotions)))


def command(stage, run, *options):
    return ['bash', str(ROOT/'run_multimodal.sh'), stage, '--run', str(run), *map(str, options)]


def execute(argv):
    print('+ '+shlex.join(argv), flush=True)
    return subprocess.run(argv, cwd=str(ROOT)).returncode


def worker(args):
    batch = args.run_dir.resolve()
    config = json.loads((batch/'batch.json').read_text())
    shard = config['shards'][args.index]
    run = batch/f'shard_{args.index:03d}'
    with lock(run/'.worker.lock'):
        def state(stage, **extra):
            atomic_json(run/'worker_status.json', dict(stage=stage, pid=os.getpid(),
                        gpu=args.gpu, updated=time.time(), **extra))
        try:
            if file_hash(Path(config['source'])) != config['source_sha256']:
                raise ValueError('Source JSON changed; do not resume this batch')
            state('prepare')
            if not (run/'plan.meta.json').exists():
                code = execute(command('prepare', run, '--source', 'psyqa', '--input', config['source'],
                                       '--offset', shard['offset'], '--limit', shard['limit'],
                                       '--seed', config['seed'], '--qa-answers', config['qa_answers'],
                                       '--invalid-record-policy', 'fail'))
                if code:
                    raise RuntimeError(f'prepare failed: {code}')
            meta = json.loads((run/'plan.meta.json').read_text())
            expected = dict(source_sha256=config['source_sha256'], seed=config['seed'],
                            offset=shard['offset'], limit=shard['limit'],
                            qa_answer_policy=config['qa_answers'], turn_limit=0)
            if any(meta.get(k) != v for k, v in expected.items()):
                raise ValueError('Existing shard plan does not match batch configuration')
            reference_maps(run)
            failures = {}
            for stage in ('audio', 'video'):
                state(stage)
                started = time.time()
                rc = execute(command(stage, run, '--gpu', args.gpu, '--keep-going',
                                     '--video-segment-seconds', config['video_segment_seconds']))
                if rc:
                    failures[stage] = rc


                    summary_path = run/(stage+'.summary.json')
                    if not summary_path.exists() or summary_path.stat().st_mtime < started:
                        raise RuntimeError(f'{stage} stopped before writing a summary: {rc}')
            state('validate')
            rc = execute(command('validate', run))
            if rc:
                failures['validate'] = rc
            if not failures:
                state('export')
                rc = execute(command('export', run))
                if rc:
                    failures['export'] = rc
            state('needs_retry' if failures else 'complete', failures=failures)
            return 1 if failures else 0
        except BaseException as error:
            state('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
            raise


def launch(args):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.session):
        raise ValueError('Session must contain only letters, digits, hyphen or underscore')
    gpus = gpu_list(args.gpus, args.instances_per_gpu)
    source = ROOT/INPUTS['psyqa']
    records = json.loads(source.read_text())
    answers = sum(len(r.get('answers', [])) if args.qa_answers == 'all' else min(1, len(r.get('answers', [])))
                  for r in records)
    batch = args.run_dir.resolve()
    config = dict(schema_version=1, source=str(source), source_sha256=file_hash(source),
                  questions=len(records), dialogue_units=answers, expected_turns=answers*2,
                  qa_answers=args.qa_answers, seed=args.seed,
                  video_segment_seconds=args.video_segment_seconds,
                  shards=partitions(len(records), len(gpus)))
    print(json.dumps(dict(run_dir=str(batch), gpu_per_instance=gpus, **config), indent=2), flush=True)
    if (batch/'batch.json').exists():
        if json.loads((batch/'batch.json').read_text()) != config:
            raise ValueError('Batch parameters changed. Keep instance count, seed and policies fixed when resuming')
    if args.dry_run:
        print('DRY RUN: no tmux, GPU, reference media access or output writes')
        return 0
    skip_review = getattr(args, 'skip_visual_review', False)
    if skip_review:
        review_status = 'skipped_by_user_request'
        print('Visual review skipped by explicit request; perceptual quality is unreviewed.', flush=True)
    else:
        gate = ROOT/'experiments/psyqa_long_video_gpu_approval.json'
        if not gate.exists():
            raise ValueError('Full generation is gated: real GPU long-video pilot and visual review are pending. '
                             'See PSYQA_PARALLEL.md, or use --skip-visual-review when explicitly requested. '
                             'No GPU process started.')
        approval = json.loads(gate.read_text())
        from psymm_generation.cli import inventory
        if (approval.get('status') != 'approved' or
                approval.get('code') != inventory()['code'] or
                approval.get('video_segment_seconds') != args.video_segment_seconds):
            raise ValueError('GPU pilot approval is absent or stale for this code/segment configuration')
        review_status = 'approved'
    batch.mkdir(parents=True, exist_ok=True)
    with lock(batch/'.launch.lock'):
        if subprocess.run(['tmux', 'has-session', '-t', '='+args.session],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            raise ValueError('tmux session already exists; attach to it instead of launching duplicate workers')
        atomic_json(batch/'batch.json', config)
        atomic_json(batch/'last_launch.json', dict(gpu_per_instance=gpus, session=args.session,
                    created_at=time.time(), launcher_sha256=file_hash(Path(__file__)),
                    visual_review=review_status))
        script = str(Path(__file__).resolve())
        monitor = shlex.join([sys.executable, script, 'status', '--run-dir', str(batch), '--watch', '5'])
        subprocess.run(['tmux', 'new-session', '-d', '-s', args.session, '-n', 'monitor', monitor], check=True)
        for i, gpu in enumerate(gpus):
            run = batch/f'shard_{i:03d}'
            run.mkdir(exist_ok=True)
            argv = [sys.executable, '-u', script, 'worker', '--run-dir', str(batch),
                    '--index', str(i), '--gpu', str(gpu)]
            shell = shlex.join(argv)+' >> '+shlex.quote(str(run/'worker.log'))+' 2>&1'
            subprocess.run(['tmux', 'new-window', '-d', '-t', args.session, '-n', f'w{i}_gpu{gpu}', shell], check=True)
            subprocess.run(['tmux', 'set-window-option', '-t', f'{args.session}:w{i}_gpu{gpu}',
                            'remain-on-exit', 'on'], check=True)
        print('Started. Attach: tmux attach -t '+args.session)
    return 0


def status(args):
    batch = args.run_dir.resolve()
    config = json.loads((batch/'batch.json').read_text())
    while True:
        totals = Counter()
        if args.watch:
            print('\033[2J\033[H', end='')
        print(time.strftime('%Y-%m-%d %H:%M:%S'), 'batch:', batch, flush=True)
        print('shard  GPU stage           planned audio_ok audio_bad video_ok video_bad')
        identities = Counter()
        for shard in config['shards']:
            run = batch/f"shard_{shard['index']:03d}"
            meta = json.loads((run/'plan.meta.json').read_text()) if (run/'plan.meta.json').exists() else {}
            state = json.loads((run/'worker_status.json').read_text()) if (run/'worker_status.json').exists() else {}
            counts = Counter()
            if (run/'state.sqlite').exists():
                try:
                    with sqlite3.connect((run/'state.sqlite').as_uri()+'?mode=ro', uri=True, timeout=2) as db:
                        counts.update({(a,b):n for a,b,n in db.execute('SELECT stage,status,COUNT(*) FROM media GROUP BY stage,status')})
                except sqlite3.Error as e:
                    print('SQLite busy/unavailable:', run.name, str(e))
            stage = state.get('stage', 'pending')
            if stage in ('prepare','audio','video','validate','export'):
                try:
                    os.kill(state['pid'], 0)
                except ProcessLookupError:
                    stage = 'stopped'
            print(f"{shard['index']:03d} {str(state.get('gpu','-')):>5} {stage:16} {meta.get('turns',0):7} "
                  f"{counts['audio','complete']:8} {counts['audio','failed']:9} "
                  f"{counts['video','complete']:8} {counts['video','failed']:9}")
            totals.update(counts)
            if (run/'reference_usage.json').exists():
                usage = json.loads((run/'reference_usage.json').read_text())
                identities.update(usage['dialogue_role_count_by_identity'])
        print(f"Expected turns: {config['expected_turns']}; audio complete: {totals['audio','complete']}; "
              f"video complete: {totals['video','complete']}; planned distinct identities: {len(identities)}")
        print('Counts are saved database states, not a fresh media integrity check. Logs: shard_*/worker.log', flush=True)
        if not args.watch:
            break
        time.sleep(args.watch)
    return 0


def collect(args):
    """Merge planned references after all shard plans exist, with no media access."""
    batch = args.run_dir.resolve()
    config = json.loads((batch/'batch.json').read_text())
    records = {}
    for shard in config['shards']:
        run = batch/f"shard_{shard['index']:03d}"
        path = run/'audio_reference_map_emo.json'
        if not path.exists():
            raise ValueError(f'Missing shard reference map: {path}')
        for asset_id, record in json.loads(path.read_text())['records'].items():
            if asset_id in records:
                raise ValueError(f'Duplicate asset ID across shards: {asset_id}')
            records[asset_id] = record
    if len(records) != config['expected_turns']:
        raise ValueError(f"Incomplete plans: {len(records)} / {config['expected_turns']} turns")
    identity_counts = Counter(record['identity'] for record in records.values())
    summary = dict(schema_version=1, batch_config_sha256=digest(config),
                   planned_turns=len(records), distinct_identities=len(identity_counts),
                   turn_count_by_identity=dict(identity_counts), records=records)
    atomic_json(batch/'psyqa_audio_reference_map_emo.json', summary)
    atomic_json(batch/'psyqa_reference_summary.json', {k:v for k,v in summary.items() if k != 'records'})
    print(json.dumps(dict(planned_turns=len(records), distinct_identities=len(identity_counts),
                          map=str(batch/'psyqa_audio_reference_map_emo.json')), ensure_ascii=False))
    return 0


def approve_long_video(args):
    """Record a human-reviewed, technically valid long-video pilot."""
    pilot = args.pilot_run.resolve()
    report_path = pilot / 'validation.json'
    if not report_path.exists():
        raise ValueError(f'Missing pilot validation report: {report_path}')
    report = json.loads(report_path.read_text())
    if report.get('failed_checks') != 0 or not report.get('records'):
        raise ValueError('Pilot validation must contain records and zero failed checks')
    video_config_path = pilot / 'video.config.json'
    if not video_config_path.exists():
        raise ValueError(f'Missing pilot video configuration: {video_config_path}')
    video_config = json.loads(video_config_path.read_text())
    if video_config.get('options', {}).get('video_segment_seconds') != args.video_segment_seconds:
        raise ValueError('Pilot segment size does not match the requested approval')
    db_path = pilot / 'state.sqlite'
    if not db_path.exists():
        raise ValueError(f'Missing pilot state database: {db_path}')
    with sqlite3.connect(str(db_path)) as db:
        completed = list(db.execute("SELECT details FROM media WHERE stage='video' AND status='complete'"))
    segmented = [json.loads(row[0]) for row in completed if json.loads(row[0]).get('segmented')]
    if not segmented:
        raise ValueError('Pilot contains no completed segmented video to review')
    evidence = args.evidence.expanduser().resolve()
    if not evidence.is_file():
        raise ValueError(f'Review evidence must be an existing file: {evidence}')
    from psymm_generation.cli import inventory
    approval = dict(schema_version=1, status='approved', approved_at=time.time(), reviewer=args.reviewer,
                    evidence=dict(path=str(evidence), sha256=file_hash(evidence)),
                    pilot_run=str(pilot), pilot_validation_sha256=file_hash(report_path),
                    video_configuration_sha256=file_hash(video_config_path),
                    completed_segmented_videos=len(segmented),
                    video_segment_seconds=args.video_segment_seconds, code=inventory()['code'])
    destination = ROOT / 'experiments/psyqa_long_video_gpu_approval.json'
    atomic_json(destination, approval)
    print(json.dumps(dict(approval=str(destination), pilot_run=str(pilot),
                          reviewed_segmented_videos=len(segmented)), ensure_ascii=False))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('launch')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--gpus', required=True)
    p.add_argument('--instances-per-gpu', type=int, default=1)
    p.add_argument('--session', default='psyqa_full')
    p.add_argument('--seed', type=int, default=20261015)
    p.add_argument('--qa-answers', choices=['first','all'], default='all')
    p.add_argument('--video-segment-seconds', type=int, choices=range(2,61), default=20)
    p.add_argument('--skip-visual-review', action='store_true',
                   help='Explicitly bypass human review; record skipped review without creating an approval')
    p.add_argument('--dry-run', action='store_true')
    p.set_defaults(func=launch)
    p = subs.add_parser('worker')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--index', type=int, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.set_defaults(func=worker)
    p = subs.add_parser('status')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--watch', type=int, default=0)
    p.set_defaults(func=status)
    p = subs.add_parser('collect')
    p.add_argument('--run-dir', type=Path, required=True)
    p.set_defaults(func=collect)
    p = subs.add_parser('approve-long-video')
    p.add_argument('--pilot-run', type=Path, required=True)
    p.add_argument('--video-segment-seconds', type=int, choices=range(2,61), required=True)
    p.add_argument('--reviewer', required=True)
    p.add_argument('--evidence', type=Path, required=True,
                   help='Existing file with the human visual-review record')
    p.set_defaults(func=approve_long_video)
    args = parser.parse_args()
    if getattr(args, 'watch', 0) < 0:
        parser.error('watch must be nonnegative')
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
