"""CPU-only dialogue identity checks for all supported dataset sources."""
import argparse
import json
import sqlite3
from pathlib import Path

from psymm_generation.core import INPUTS, atomic_json, digest

SOURCES = frozenset(INPUTS)


def validate_items(items):
    dialogues, assets, sources = {}, set(), set()
    for item in items:
        asset, dialogue, role = item['asset_id'], item['dialogue_id'], item['role']
        source = item.get('source')
        if source not in SOURCES or role not in ('seeker', 'supporter'):
            raise ValueError(f'Invalid dialogue source/role: {asset}')
        sources.add(source)
        if asset in assets:
            raise ValueError(f'Duplicate dialogue asset: {asset}')
        assets.add(asset)
        ref = item['reference']
        identity, segment = str(ref['identity']), ref['segment']
        if not segment.endswith('_' + identity):
            raise ValueError(f'Reference segment/profile identity mismatch: {asset}')
        for field in ('audio', 'video', 'portrait'):
            if Path(ref[field]).stem != segment:
                raise ValueError(f'Unpaired reference {field}: {asset}; expected segment {segment}')
        if ref.get('dialogue_scope') != f'{dialogue}:{role}':
            raise ValueError(f'Reference assigned to another dialogue/role: {asset}')
        expected_emotion = None if item['emotion'] == 'neutral' else item['emotion']
        if ref.get('reference_emotion') != expected_emotion:
            raise ValueError(f'Reference emotion mismatch: {asset}')
        binding = dict(identity=identity, segment=segment,
                       reference_audio=ref['audio'], audio_sha256=ref['audio_sha256'],
                       reference_image=ref['portrait'], portrait_sha256=ref['portrait_sha256'],
                       reference_video=ref['video'], emotion=item['emotion'])
        entry = dialogues.setdefault(dialogue, dict(source=source, source_index=item['source_index'],
                                    answer_index=item.get('answer_index'), roles={}))
        if (entry['source'], entry['source_index'], entry['answer_index']) != (
                source, item['source_index'], item.get('answer_index')):
            raise ValueError(f'Dialogue ID reused for another source record/answer: {asset}')
        previous = entry['roles'].setdefault(role, binding)
        if previous != binding:
            raise ValueError(f'Identity/reference changes within dialogue role: {dialogue}/{role}')
    if not assets:
        raise ValueError('Dialogue identity plan is empty')
    if len(sources) != 1:
        raise ValueError('A run must contain exactly one dataset source')
    for dialogue, entry in dialogues.items():
        roles = entry['roles']
        if set(roles) != {'seeker', 'supporter'}:
            raise ValueError(f'Dialogue requires both roles: {dialogue}')
        if roles['seeker']['identity'] == roles['supporter']['identity']:
            raise ValueError(f'Dialogue requires two distinct identities: {dialogue}')
    source = next(iter(sources))
    return dict(schema_version=1, source=source,
                scope='independent_question_answer' if source == 'psyqa' else 'whole_dialogue',
                turns=len(assets), dialogue_count=len(dialogues), dialogues=dialogues)


def check_maps(run, items):
    for name, field, expected_hash in (
            ('audio_reference_map_emo.json', 'reference_audio', 'audio_sha256'),
            ('ref_map.json', 'reference_image', 'portrait_sha256')):
        path = run/name
        if not path.exists():
            continue
        mapping = json.loads(path.read_text())
        if mapping.get('schema_version') != 1 or 'records' not in mapping:
            raise ValueError(f'Legacy identity map cannot be used with this run: {path}')
        records = mapping['records']
        if set(records) != {item['asset_id'] for item in items}:
            raise ValueError(f'Identity map does not cover exactly this plan: {path}')
        for item in items:
            ref, record = item['reference'], records[item['asset_id']]
            reference_key = 'audio' if field == 'reference_audio' else 'portrait'
            expected = dict(identity=ref['identity'], segment=ref['segment'],
                            role=item['role'], emotion=item['emotion'],
                            **{field: ref[reference_key], 'sha256': ref[expected_hash]})
            if any(record.get(k) != value for k, value in expected.items()):
                raise ValueError(f'Identity map disagrees with plan: {path.name}/{item["asset_id"]}')


def check_audio_receipts(run, items):
    path = run/'state.sqlite'
    if not path.exists():
        return 0
    checked = 0
    with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True, timeout=10) as db:
        receipts = {asset: json.loads(details) for asset, details in db.execute(
            "SELECT asset,details FROM media WHERE stage='audio' AND status='complete'")}
    for item in items:
        receipt = receipts.get(item['asset_id'])
        if receipt is None:
            continue
        ref = item['reference']
        if (receipt.get('reference_audio') != ref['audio'] or
                receipt.get('reference_audio_sha256') != ref['audio_sha256']):
            raise ValueError(f'Generated audio uses another identity/reference: {item["asset_id"]}; rerun audio')
        checked += 1
    return checked


def _check_run(run, stage='plan'):
    from psymm_generation.cli import read_plan
    run = Path(run).resolve()
    items = read_plan(run)
    manifest = validate_items(items)
    check_maps(run, items)
    audio_receipts = check_audio_receipts(run, items) if stage in ('video', 'validate', 'export') else 0
    atomic_json(run/'dialogue_identity_map.json', manifest)
    report = dict(status='passed', stage=stage, scope=manifest['scope'],
                  turns=manifest['turns'], dialogues=manifest['dialogue_count'],
                  completed_audio_references_checked=audio_receipts,
                  identity_manifest_sha256=digest(manifest),
                  checks='Distinct fixed role identities; same-segment audio/portrait; '
                         'plan/map agreement; completed audio reference agreement before video. '
                         'Metadata checks do not measure generated face or voice similarity.')
    atomic_json(run/'identity_validation.json', report)
    print(f"IDENTITY CHECK PASSED: {report['dialogues']} {manifest['source']} dialogues, "
          f"{report['turns']} turns, {audio_receipts} completed audio references checked", flush=True)
    return report


def check_run(run, stage='plan'):
    run = Path(run).resolve()
    try:
        return _check_run(run, stage)
    except (ValueError, KeyError, OSError, sqlite3.Error) as error:
        if run.is_dir():
            atomic_json(run/'identity_validation.json',
                        dict(status='failed', stage=stage, error=str(error)))
        raise


def preflight(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('stage', nargs='?')
    parser.add_argument('--run', type=Path)
    args, _ = parser.parse_known_args(argv)
    if args.stage not in ('audio', 'video', 'validate', 'export') or args.run is None:
        return
    path = args.run/'plan.jsonl'
    if not path.exists():
        return
    check_run(args.run, args.stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--stage', choices=['plan', 'audio', 'video', 'validate', 'export'], default='plan')
    args = parser.parse_args()
    check_run(args.run, args.stage)
