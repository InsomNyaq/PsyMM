"""Build text response training rows from verified generated seeker media."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from psymm_generation.cli import database, inventory, read_plan, run_lock, verified_record
from psymm_generation.core import atomic_json, digest, file_hash, probe_audio, probe_video


def response_rows(items):
    groups = defaultdict(list)
    for item in items:
        groups[item['dialogue_id']].append(item)
    rows, skipped = [], []
    for dialogue_id, turns in sorted(groups.items()):
        turns.sort(key=lambda x: x['turn_index'])
        for i, turn in enumerate(turns):
            if turn['role'] != 'supporter':
                continue
            if i == 0 or turns[i-1]['role'] != 'seeker':
                skipped.append(dict(asset_id=turn['asset_id'], reason='No immediately preceding seeker'))
                continue
            seeker = turns[i-1]

            history = '\n'.join(f"{t['role']}: {t['text']}" for t in turns[:i])
            rows.append(dict(id=turn['asset_id'], dialogue_id=dialogue_id,
                source=turn['source'], source_id=turn['source_id'],
                input_asset_id=seeker['asset_id'], target_asset_id=turn['asset_id'],
                audio=seeker['media']['audio']['path'], video=seeker['media']['video']['path'],
                intended_input_emotion=seeker['emotion'], conversations=[
                    dict(from_='human',value='<speech><video>\n'+history+
                         '\nRespond as an empathetic supporter. Output only the next response.'),
                    dict(from_='gpt',value=turn['text'])]))
            for conversation in rows[-1]['conversations']:
                conversation['from'] = conversation.pop('from_')
    return rows, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output; training JSON is never overwritten')
    run = args.run.resolve()
    with run_lock(run):
        manifest = run/'multimodal_manifest.json'
        items = json.loads(manifest.read_text(encoding='utf-8'))
        plan = read_plan(run)
        if len(plan) != len(items) or [p['asset_id'] for p in plan] != [p['asset_id'] for p in items]:
            raise ValueError('Manifest does not match the prepared run')
        db, info = database(run), inventory()
        for item, planned in zip(items,plan):
            if any(item.get(k) != v for k,v in planned.items()):
                raise ValueError('Manifest plan fields changed')
            for stage in ('audio','video'):
                row = verified_record(db,run,planned,stage,info)
                if row[0] != item['media'][stage]['path'] or row[1] != item['media'][stage]['sha256']:
                    raise ValueError('Manifest media mapping changed')
            aq = probe_audio(item['media']['audio']['path'])
            probe_video(item['media']['video']['path'],aq['seconds'],True)
        db.close()
        rows, skipped = response_rows(items)
        if not rows:
            raise ValueError('No valid seeker-to-supporter training examples')
        atomic_json(args.output,rows)
        atomic_json(args.output.with_suffix('.provenance.json'),dict(
            manifest=str(manifest),manifest_sha256=file_hash(manifest),
            examples=len(rows),skipped=skipped,
            split_policy='Unsplit. Assign whole source dialogues to train/validation/test before training.',
            perceptual_review_status='pending; technical verification is not subjective quality approval'))
        print(f'Built {len(rows)} response rows; {len(skipped)} unsupported response transitions recorded')


if __name__ == '__main__':
    main()
