#!/usr/bin/env python3
import sys

from psymm_generation.cli import main as cli_main
from dialogue_identity import preflight


def main():
    preflight(sys.argv[1:])
    prepare_extes_nonempty_input()
    return cli_main()


def prepare_extes_nonempty_input():
    """Skip empty ExTES speech and audit the authorized missing-role quarantine.

    The derived input belongs to the new run. Inference code and fingerprints
    remain unchanged, so unrelated datasets' valid media remain resumable.
    """
    import argparse
    import json
    from pathlib import Path
    from psymm_generation.core import (ROOT, INPUTS, ROLES, atomic_json, clean_text,
                                      file_hash, normalized_dialogue)
    from psymm_generation.cli import run_lock
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('stage', nargs='?')
    parser.add_argument('--source')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--run', type=Path)
    args, _ = parser.parse_known_args(sys.argv[1:])
    if (args.stage != 'prepare' or args.source != 'extes' or args.run is None or
            '--help' in sys.argv or '-h' in sys.argv):
        return
    source = (args.input or ROOT / INPUTS['extes']).resolve()
    run = args.run.resolve()
    data = json.loads(source.read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError('ExTES input must be a list of dialogue records')
    skipped, index_maps, quarantined = [], [], []
    for index, record in enumerate(data):
        key = 'dialog' if 'dialog' in record else 'messages'
        turns = record.get(key)
        if not isinstance(turns, list):
            continue
        retained, original_indices = [], []
        for turn_index, turn in enumerate(turns):
            role = turn.get('speaker', turn.get('role', turn.get('from')))
            text = turn.get('content', turn.get('text', turn.get('value')))
            empty = False
            if role in ROLES and isinstance(text, str):
                try:
                    clean_text(text)
                except ValueError:
                    empty = True
            if empty:
                skipped.append(dict(source_index=index, original_turn_index=turn_index,
                                    original_role=role, original_text=text))
            else:
                retained.append(turn)
                original_indices.append(turn_index)
        if len(retained) != len(turns):
            record[key] = retained
            index_maps.append(dict(source_index=index,
                                   retained_original_turn_indices=original_indices))
            present_roles = {ROLES[t.get('speaker', t.get('role', t.get('from')))]
                             for t in retained
                             if t.get('speaker', t.get('role', t.get('from'))) in ROLES}
            if present_roles != {'seeker', 'supporter'}:
                quarantined.append(dict(source_index=index,
                    reason='Missing a speaking role after empty utterances were removed',
                    missing_roles=sorted({'seeker', 'supporter'} - present_roles),
                    original_record=dict(record, **{key: turns})))


                record[key] = []
    excluded = {r['source_index'] for r in quarantined}
    for index, record in enumerate(data):
        if index not in excluded:


            list(normalized_dialogue('extes', record, index))
    with run_lock(run):
        if (run/'plan.meta.json').exists() or (run/'state.sqlite').exists():
            raise ValueError('Run already prepared; resume audio/video or choose a new --run')
        derived = run/'source_nonempty.json'
        atomic_json(derived, data)
        atomic_json(run/'empty_turns_skipped.json', dict(
            schema_version=1, policy='Skip empty ExTES speech; quarantine dialogues missing a speaking role',
            original_source=str(source), original_sha256=file_hash(source),
            derived_source=str(derived), derived_sha256=file_hash(derived),
            empty_turn_count=len(skipped),
            skipped_turn_count=sum(r['source_index'] not in excluded for r in skipped),
            affected_dialogues=len(index_maps), quarantined_dialogues=len(quarantined),
            skipped=skipped, turn_index_maps=index_maps))
        atomic_json(run/'quarantined_source_records.json', dict(
            policy='User authorized quarantine of dialogues missing a role after skipping empty speech',
            original_source=str(source), records=quarantined))

    if '--input' in sys.argv:
        at = sys.argv.index('--input')
        sys.argv[at + 1] = str(derived)
    else:
        sys.argv[:] = [x for x in sys.argv if not x.startswith('--input=')]
        sys.argv.extend(['--input', str(derived)])
    if '--invalid-record-policy' in sys.argv:
        at = sys.argv.index('--invalid-record-policy')
        sys.argv[at + 1] = 'quarantine'
    else:
        sys.argv[:] = [x for x in sys.argv if not x.startswith('--invalid-record-policy=')]
        sys.argv.extend(['--invalid-record-policy', 'quarantine'])
    retained_skips = sum(r['source_index'] not in excluded for r in skipped)
    print(f'ExTES: found {len(skipped)} empty turns; skipped {retained_skips} in retained dialogues; '
          f'quarantined {len(quarantined)} missing-role dialogues; '
          f'audit: {run / "empty_turns_skipped.json"}', flush=True)


def dataset_main(stage, source):
    """Dataset entry points use the same checks as run_multimodal.sh."""
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--run', type=Path)
    args, _ = parser.parse_known_args(sys.argv[1:])
    if args.run is not None and (args.run / 'plan.jsonl').exists():
        with (args.run / 'plan.jsonl').open(encoding='utf-8') as handle:
            first = next((json.loads(line) for line in handle if line.strip()), {})
        if first.get('source') != source:
            raise ValueError(f'{source} entry cannot use a {first.get("source")} plan')
    sys.argv.insert(1, stage)
    return main()


if __name__ == '__main__':
    raise SystemExit(main())
