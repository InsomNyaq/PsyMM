"""CPU metadata audit and group-isolated revision split. No media/model reads.

Never replaces historical assignments or claims they were used in old training.
Only standard-library imports and generation normalization are used.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from psymm_generation.core import INPUTS, EMOTIONS, SUPPORTER_EMOTIONS, normalized_dialogue

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def canonical(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())

def group_keys(source, record, dialogues):
    keys = []
    for name in ('conversation_id', 'dialogueID', 'post_id', 'questionID'):
        if name in record and record[name] is not None and record[name] != '':
            keys.append('id:' + digest([source, name, record[name]]))
            break
    for dialogue in dialogues:
        turns = [(t['role'], canonical(t['text'])) for t in dialogue['turns']]
        keys.append('dialogue:' + digest(turns))
        seekers = [text for role, text in turns if role == 'seeker']


        if source in ('psyqa', 'psych8k') or sum(map(len, seekers)) >= 80:
            keys.append('seeker_context:' + digest(seekers))
        if seekers and len(seekers[0]) >= 80:
            keys.append('first_seeker:' + digest(seekers[0]))
    return sorted(set(keys))

def split_for(group, seed):
    value = int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:16], 16)

    if value < (2**64 * 80) // 100:
        return 'train'
    if value < (2**64 * 90) // 100:
        return 'validation'
    return 'test'

class UnionFind:
    def __init__(self):
        self.parent = []
    def add(self):
        self.parent.append(len(self.parent))
    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i
    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)

def grouped_assignments(nodes, seed):
    """Connect shared keys transitively; membership is independent of row order."""
    uf, first_key = UnionFind(), {}
    for i, node in enumerate(nodes):
        if not node['keys']:
            raise ValueError('Every valid source record needs a grouping key')
        uf.add()
        for key in node['keys']:
            if key in first_key:
                uf.join(i, first_key[key])
            else:
                first_key[key] = i
    groups = defaultdict(list)
    for i in range(len(nodes)):
        groups[uf.find(i)].append(i)
    for indices in groups.values():
        all_keys = sorted({k for i in indices for k in nodes[i]['keys']})
        group_id = digest(all_keys)
        yield indices, all_keys, group_id, split_for(group_id, seed)

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261018)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if (out/'audit_summary.json').exists() or (out/'revision_split_assignments.jsonl').exists():
        raise FileExistsError('Use a new output directory for a new audit')
    old = ROOT/'runs/experiment_freeze_20261015'
    frozen = json.loads((old/'freeze_manifest.json').read_text())
    old_assignments = {}
    with (old/'split_assignments.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            for r in row['records']:
                old_assignments[(row['source'], r['source_index'])] = row['split']
    nodes = []
    source_results, rejected, metrics = {}, [], {}
    for source, relative in INPUTS.items():
        path = ROOT/relative
        records = json.loads(path.read_text(encoding='utf-8'))
        counts = Counter()
        raw = Counter()
        gt_pred, record_keys = [], Counter()
        for index, record in enumerate(records):
            record_keys.update(record.keys())
            gt = str(record.get('emotion_type', '')).strip().lower()
            pred = str(record.get('emotion_annotation', '')).strip().lower()
            if gt and pred:
                gt_pred.append((gt, pred))
            try:
                dialogues = list(normalized_dialogue(source, record, index))
            except (ValueError, KeyError, TypeError) as exc:
                rejected.append(dict(source=source, source_index=index,
                                     record_sha256=digest(record), reason=str(exc)))
                continue
            keys = group_keys(source, record, dialogues)
            counts['valid_records'] += 1
            counts['dialogues_first_answer'] += len(dialogues)
            if source == 'psyqa':
                counts['answers_available'] += len(record.get('answers', []))
            for dialogue in dialogues:
                role_labels = [{t['emotion'] for t in dialogue['turns'] if t['role'] == role}
                               for role in ('seeker','supporter')]
                counts['dialogues_incompatible_with_fixed_role_anchor'] += any(len(v)>1 for v in role_labels)
                for turn in dialogue['turns']:
                    counts['turns'] += 1
                    counts['role:' + turn['role']] += 1
                    counts['origin:' + turn['emotion_origin']] += 1
                    counts['emotion:' + turn['emotion']] += 1
                    counts['language:' + turn['language']] += 1
                    raw[str(turn['raw_emotion'])] += 1
            nodes.append(dict(source=source, source_index=index,
                              record_sha256=digest(record), keys=keys,
                              dialogue_ids=[d['dialogue_id'] for d in dialogues],
                              dialogues=len(dialogues), turns=sum(len(d['turns']) for d in dialogues),
                              historical_split=old_assignments.get((source,index))))
        source_results[source] = dict(path=str(path.resolve()), sha256=file_hash(path),
            historical_hash_matches=file_hash(path) == frozen['sources'][source]['sha256'],
            records=len(records), counts=dict(counts), raw_emotions=dict(raw),
            top_level_fields=dict(record_keys))
        print(f'{source}: {len(records)} records, {counts["valid_records"]} valid', flush=True)
        metrics[source] = dict(paired_records=len(gt_pred), correct=sum(g==p for g,p in gt_pred),
            accuracy=sum(g==p for g,p in gt_pred)/len(gt_pred) if gt_pred else None,
            interpretation='Top-level emotion_type vs emotion_annotation only; not independent human turn validation')
    groups = list(grouped_assignments(nodes, args.seed))
    old_cross, split_counts = [], defaultdict(Counter)
    key_splits = defaultdict(set)
    with (out/'revision_split_assignments.jsonl').open('w', encoding='utf-8') as handle:
        for indices, all_keys, group_id, split in groups:
            previous = {nodes[i]['historical_split'] for i in indices} - {None}
            if len(previous) > 1:
                old_cross.append(dict(group_id=group_id, historical_splits=sorted(previous),
                                      records=[{k:nodes[i][k] for k in ('source','source_index')} for i in indices]))
            for key in all_keys:
                key_splits[key].add(split)
            for i in indices:
                node = nodes[i]
                split_counts[node['source']][split + ':records'] += 1
                split_counts[node['source']][split + ':dialogues'] += node['dialogues']
                split_counts[node['source']][split + ':turns'] += node['turns']
                row = {k:v for k,v in node.items() if k != 'keys'}
                row.update(group_id=group_id, split=split)
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    violations = sum(len(v)>1 for v in key_splits.values())
    assert violations == 0
    write(out/'quarantined_records.json', rejected)
    write(out/'historical_cross_split_duplicates.json', old_cross)

    run_audits = []
    for run in sorted((ROOT/'runs').iterdir()):
        if not run.is_dir() or not (run/'plan.meta.json').is_file():
            continue
        meta = json.loads((run/'plan.meta.json').read_text())
        counter, reference_checks = Counter(), Counter()
        identities = defaultdict(lambda: defaultdict(set))
        anchors = defaultdict(lambda: defaultdict(set))
        role_emotions = defaultdict(lambda: defaultdict(set))
        with (run/'plan.jsonl').open() as handle:
            for line in handle:
                row = json.loads(line)
                ref = row['reference']
                counter[row['role']+':'+row['emotion']] += 1
                identities[row['dialogue_id']][row['role']].add(ref['identity'])
                anchors[row['dialogue_id']][row['role']].add(ref['segment'])
                role_emotions[row['dialogue_id']][row['role']].add(row['emotion'])
                if row['emotion'] == 'neutral':
                    reference_checks['neutral_with_pool'] += ref.get('reference_emotion') is not None
                else:
                    if 'reference_emotion' not in ref:
                        reference_checks['pool_emotion_unverifiable_legacy_metadata'] += 1
                    else:
                        reference_checks['pool_emotion_mismatch'] += ref['reference_emotion'] != row['emotion']
        for did, roles in identities.items():
            reference_checks['within_role_identity_changes'] += sum(len(ids)>1 for ids in roles.values())
            reference_checks['roles_share_identity'] += bool(roles.get('seeker',set()) & roles.get('supporter',set()))
            reference_checks['within_role_anchor_changes'] += sum(len(ids)>1 for ids in anchors[did].values())
            reference_checks['roles_with_multiple_emotions'] += sum(len(v)>1 for v in role_emotions[did].values())
        states = []
        if (run/'state.sqlite').is_file():
            db = sqlite3.connect((run/'state.sqlite').as_uri()+'?mode=ro', uri=True)
            try:
                tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                if 'media' in tables:
                    states = db.execute('SELECT stage,status,count(*) FROM media GROUP BY stage,status').fetchall()
                else:
                    states = {'unrecognized_tables': tables}
            finally:
                db.close()
        configs = []
        for path in sorted((run/'configurations').glob('*.json')):
            value = json.loads(path.read_text())
            configs.append(dict(path=str(path), sha256=file_hash(path), configuration=value))
        run_audits.append(dict(run=str(run), meta=meta,
            actual_plan_sha256=file_hash(run/'plan.jsonl'),
            plan_hash_matches=file_hash(run/'plan.jsonl') == meta['plan_sha256'],
            counts=dict(counter), reference_checks=dict(reference_checks),
            states=states, configurations=configs,
            limitation='Metadata consistency only; perceptual emotion/quality not evaluated'))
    write(out/'generation_runs.json', run_audits)

    from experiments.model_setup.checkpoint_paths import CHECKPOINT_GROUPS, get_checkpoint
    checkpoints = []
    for group, spec in CHECKPOINT_GROUPS.items():
        print(f'Hashing registered adapters: {group}', flush=True)
        for step in spec['steps']:
            adapter, base = get_checkpoint(group, step)
            cfg = json.loads((adapter/'adapter_config.json').read_text())
            item = dict(group=group, step=step, adapter=str(adapter), base=str(base),
                        adapter_configuration=cfg, files={})
            for name in ('adapter_config.json','args.json','trainer_state.json','adapter_model.safetensors','adapter_model.bin'):
                p = adapter/name
                if not p.is_file():
                    continue
                try:
                    item['files'][name] = dict(bytes=p.stat().st_size, sha256=file_hash(p), readable=True)
                except PermissionError as exc:
                    item['files'][name] = dict(bytes=p.stat().st_size, sha256=None, readable=False,
                                              error=f'{type(exc).__name__}: {p}')
                    continue
                if name == 'args.json':
                    a = json.loads(p.read_text())
                    item['training_arguments'] = {k:v for k,v in a.items() if k in {
                        'model','model_type','dataset','val_dataset','split_dataset_ratio','seed','data_seed',
                        'learning_rate','max_steps','num_train_epochs','max_length','lora_rank','lora_alpha',
                        'target_modules','eval_strategy','evaluation_strategy','save_steps','save_strategy',
                        'gradient_accumulation_steps','per_device_train_batch_size','train_type','tuner_type',
                        'freeze_llm','freeze_vit','freeze_aligner','load_best_model_at_end','metric_for_best_model',
                        'system','template','use_audio_in_video','resume_from_checkpoint',
                        'max_new_tokens','temperature','top_p','top_k','num_beams','truncation_strategy',
                        'max_pixels','adapters','dataset_shuffle','dataset_num_proc','strict','swift_version'}}
                if name == 'trainer_state.json':
                    a = json.loads(p.read_text())
                    item['trainer_state'] = {k:a.get(k) for k in ('global_step','best_model_checkpoint','best_metric','max_steps')}
                    item['logged_evaluation_entries'] = sum(any(k.startswith('eval_') for k in v) for v in a.get('log_history',[]))
            checkpoints.append(item)
    write(out/'checkpoint_provenance.json', checkpoints)
    write(out/'emotion_policy.json', dict(mapping=EMOTIONS, supporter_defaults=SUPPORTER_EMOTIONS,
        neutral='Default TTS expression with identity conversion; no neutral emotion-pool sampling',
        note='Generation defaults are not independently validated emotion labels'))
    tracked = git('ls-files').splitlines()
    code = {name:file_hash(ROOT/name) for name in tracked if (ROOT/name).is_file()}
    write(out/'tracked_file_hashes.json', code)
    summary = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(), git_commit=git('rev-parse','HEAD'),
        git_status=git('status','--porcelain','--untracked-files=no'),
        audit_script_sha256=file_hash(__file__), sources=source_results,
        historical_annotation_comparison=metrics, groups=len(groups), valid_records=len(nodes),
        duplicate_groups=sum(len(g[0])>1 for g in groups),
        max_group_records=max(len(g[0]) for g in groups),
        historical_cross_split_duplicate_groups=len(old_cross),
        historical_cross_split_duplicate_records=sum(len(g['records']) for g in old_cross),
        historical_split_probability=dict(train=216/256, validation=20/256, test=20/256),
        revision_policy=dict(seed=args.seed, ratios=[80,10,10], unit='group of original source records',
            group_keys=['source-scoped original ID','exact normalized full dialogue',
                        'identical seeker trajectory (QA always; multi-turn >=80 characters)',
                        'identical first seeker text >=80 characters'],
            normalization='Unicode NFKC, casefold, whitespace collapse; exact matching only',
            missing='Fuzzy/paraphrased duplicates and historical training exposure require separate checks',
            qa_answers='first; all answers of the question inherit the same split',
            historical_warning='New revision split does not establish old checkpoint train/test isolation'),
        revision_cross_split_key_violations=violations, split_counts={k:dict(v) for k,v in split_counts.items()},
        quarantined_records=len(rejected), registered_checkpoints=len(checkpoints),
        unreadable_checkpoint_files=[dict(group=c['group'], step=c['step'], file=name)
            for c in checkpoints for name,v in c['files'].items() if not v['readable']],
        gpu_used=False, media_opened=False,
        artifacts={p.name:file_hash(p) for p in out.iterdir() if p.is_file()})
    write(out/'audit_summary.json', summary)
    print(json.dumps({k:summary[k] for k in ('valid_records','groups','duplicate_groups',
        'historical_cross_split_duplicate_groups','revision_cross_split_key_violations',
        'quarantined_records','registered_checkpoints')}, indent=2))

if __name__ == '__main__':
    main()
