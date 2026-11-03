"""CPU-only E1 audit, deterministic sampling, blind annotation and scoring.

Run from experiments/ in PsyMM. Standard library only; no model/media loading.
Existing source records and generator code are never modified.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
import math
from pathlib import Path
import random
import subprocess
import sys

CORE = ('angry', 'fear', 'sad', 'happy', 'disgusted', 'surprised', 'contempt', 'neutral')
ALIASES = {'anger': 'angry', 'sadness': 'sad', 'happiness': 'happy',
           'joy': 'happy', 'surprise': 'surprised', 'disgust': 'disgusted',
           'none': 'neutral', 'anxious': 'anxiety', 'depressed': 'depression'}
FINE_MAP = {
    'angry': 'angry', 'annoyance': 'angry', 'jealousy': 'angry',
    'fear': 'fear', 'anxiety': 'fear', 'panic': 'fear', 'paranoia': 'fear',
    'nervousness': 'fear', 'sad': 'sad', 'depression': 'sad',
    'emptiness': 'sad', 'hopelessness': 'sad', 'loneliness': 'sad',
    'grief': 'sad', 'shame': 'sad', 'guilt': 'sad', 'pain': 'sad',
    'happy': 'happy', 'surprised': 'surprised', 'disgusted': 'disgusted',
    'contempt': 'contempt', 'neutral': 'neutral',
}
SPECIAL = ('other', 'uncertain')
LANGUAGES = ('zh', 'en', 'mixed', 'other', 'uncertain')
FIELDS = ('fine', 'core', 'language')


def canonical(value):
    key = str(value).strip().lower()
    return ALIASES.get(key, key)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def metrics(pairs, labels):
    """Weighted confusion/F1; zero-support classes retained and identified.

    pairs = (gold, predicted, positive_weight). Macro over fixed taxonomy and
    over gold-supported classes are both reported; no predicted-only omission.
    """
    labels = list(labels)
    cm = {g: {p: 0.0 for p in labels} for g in labels}
    for g, p, w in pairs:
        if g not in cm or p not in cm or not math.isfinite(w) or w <= 0:
            raise ValueError('Invalid class or weight')
        cm[g][p] += w
    support = {g: sum(cm[g].values()) for g in labels}
    total = sum(support.values())
    per_class = {}
    for k in labels:
        tp, s = cm[k][k], support[k]
        predicted = sum(cm[g][k] for g in labels)
        precision = tp / predicted if predicted else 0.0
        recall = tp / s if s else 0.0
        f1 = 2 * tp / (s + predicted) if s + predicted else 0.0
        per_class[k] = dict(support=s, predicted_support=predicted,
                            precision=precision, recall=recall, f1=f1)
    supported = [k for k in labels if support[k] > 0]
    result = dict(n=len(pairs), weight_sum=total,
                accuracy=sum(cm[k][k] for k in labels) / total if total else None,
                macro_f1_fixed=sum(per_class[k]['f1'] for k in labels) / len(labels),
                macro_f1_gold_supported=(sum(per_class[k]['f1'] for k in supported) /
                                         len(supported) if supported else None),
                supported_classes=supported, per_class=per_class, confusion=cm)
    if set(CORE).issubset(labels):
        result['macro_f1_core8'] = sum(per_class[k]['f1'] for k in CORE) / 8
        result['macro_f1_emotion7'] = sum(per_class[k]['f1'] for k in CORE if k != 'neutral') / 7
    return result


def kappa(a, b):
    if not a:
        return None
    ca, cb, n = Counter(a), Counter(b), len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys()) / n ** 2
    return dict(n=n, agreement=observed,
                cohen_kappa=(observed - expected) / (1 - expected) if expected < 1 else None)


def select_frame(rows, seed):
    """One hash-selected representative per global E0 group; source then fixed.

    This defines a representative population, not the original record population.
    Hash selection is independent of the natural sample and of emotion labels.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[row['group_id']].append(row)
    result = []
    for gid in sorted(groups):
        result.append(min(groups[gid], key=lambda r: digest(
            [seed, 'representative', gid, r['source'], r['source_index']])))
    return result


def select_samples(frame, seed, n, tail_min):
    selected, summaries = [], {}
    by_source = defaultdict(list)
    for row in frame:
        by_source[row['source']].append(row)
    for source, population in sorted(by_source.items()):
        population = sorted(population, key=lambda r: r['group_id'])
        count = min(n, len(population))
        rng = random.Random(digest([seed, source, 'natural']))
        natural = rng.sample(population, count)
        natural_ids = {r['group_id'] for r in natural}
        tails = []
        for label in CORE:
            if label == 'neutral':
                continue
            have = sum(r['predicted_core'] == label for r in natural)
            eligible = [r for r in population if r['group_id'] not in natural_ids
                        and r['predicted_core'] == label]
            tail_rng = random.Random(digest([seed, source, label, 'tail']))
            tails.extend(tail_rng.sample(eligible, min(max(0, tail_min - have), len(eligible))))
        for cohort, part in [('natural', natural), ('tail', tails)]:
            for record in part:
                row = dict(record, cohort=cohort)

                row['group_inclusion_probability'] = count / len(population) if cohort == 'natural' else None
                row['target_conditional_probability'] = 1 / row['eligible_turns']
                row['turn_inclusion_probability'] = (count / (len(population) * row['eligible_turns'])
                                                      if cohort == 'natural' else None)
                row['group_weight'] = len(population) / count if cohort == 'natural' else None
                row['turn_weight'] = (len(population) * row['eligible_turns'] / count
                                      if cohort == 'natural' else None)
                selected.append(row)
        summaries[source] = dict(groups=len(population), natural=len(natural), tail=len(tails),
            tail_fraction=len(tails)/(len(natural)+len(tails)),
            population_core=dict(Counter(r['predicted_core'] for r in population)),
            natural_core=dict(Counter(r['predicted_core'] for r in natural)),
            tail_core=dict(Counter(r['predicted_core'] for r in tails)))
    assert len({r['group_id'] for r in selected}) == len(selected)
    return selected, summaries


def public_tasks(sample, dialogues):
    tasks = []
    for row in sample:
        dialogue = dialogues[(row['source'], row['source_index'])]
        turns = dialogue['turns']

        public = lambda t: dict(role=t['role'], text=t['text'])
        target_pos = row['target_position']
        for scope in ('target', 'dialogue'):
            context = turns[:target_pos + 1] if scope == 'target' else turns
            tasks.append(dict(task_id=row['sample_id'] + ':' + scope, scope=scope,
                              target_position=target_pos if scope == 'target' else None,
                              context=[public(t) for t in context]))
    return tasks


def render_html(path, tasks, package):

    embedded = json.dumps(dict(package_id=package, tasks=tasks), ensure_ascii=False).replace('<', '\\u003c')
    options = dict(fine=sorted(FINE_MAP) + list(SPECIAL), core=list(CORE) + list(SPECIAL),
                   language=list(LANGUAGES))
    page = r'''<!doctype html><meta charset="utf-8"><title>E1 独立标注</title>
<style>body{font:17px system-ui;max-width:1000px;margin:24px auto;padding:15px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f4f4;padding:15px}select,input,button{font:inherit;margin:8px;padding:6px}textarea{width:95%;height:65px}label{display:block}.seeker{border-left:5px solid #5588aa}.supporter{border-left:5px solid #999}</style>
<h1>E1 独立情绪标注</h1><p>请先阅读配套指南。先完成全部 target，再进入 dialogue。判断文本中 seeker 的情绪；不要推测语音或表情。细类和核心类别分别独立判断；允许不符合既定映射。other=分类表外，uncertain=信息不足。请勿与其他标注者讨论或查看管理员文件。</p>
<label>匿名标注者 ID <input id="rater" placeholder="如 R01"></label>
<p id="status"></p><button id="prev">上一条</button><button id="next">下一条</button><button id="download">导出 JSON（未完成也可保存）</button><p>进度保存在本浏览器；换电脑前请导出并用下方文件框恢复。导出文件包含所选标签和匿名 ID。</p><input type="file" id="restore" accept=".json"><h2 id="heading"></h2><div id="context"></div>
<label>细粒度情绪 <select id="fine"></select></label><label>核心情绪 <select id="core"></select></label><label>文本语言 <select id="language"></select></label><label>简短理由（尤其 other/uncertain/混合情绪）<textarea id="note"></textarea></label>
<script type="application/json" id="payload">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('payload').textContent), options=OPTIONS;
const key='PsyMM-E1-'+data.package_id+'-'+location.pathname;let saved=JSON.parse(localStorage.getItem(key)||'null')||{rater_id:'',annotations:[]};
let answers=Object.fromEntries(saved.annotations.map(r=>[r.task_id,r])), pos=0;
const el=id=>document.getElementById(id), complete=t=>options.fine.includes(answers[t.task_id]?.fine)&&options.core.includes(answers[t.task_id]?.core)&&options.language.includes(answers[t.task_id]?.language);
el('rater').value=saved.rater_id;
for(const field of Object.keys(options)){let blank=document.createElement('option');blank.value='';blank.textContent='请选择';el(field).append(blank);for(const value of options[field]){let o=document.createElement('option');o.value=value;o.textContent=value;el(field).append(o);}}
function persist(){const t=data.tasks[pos];answers[t.task_id]={task_id:t.task_id,fine:el('fine').value,core:el('core').value,language:el('language').value,note:el('note').value};localStorage.setItem(key,JSON.stringify({rater_id:el('rater').value,annotations:Object.values(answers)}));}
function show(){const t=data.tasks[pos],a=answers[t.task_id]||{};el('status').textContent=`${pos+1}/${data.tasks.length}；已完整标注 ${data.tasks.filter(complete).length}；ID ${t.task_id}`;el('heading').textContent=t.scope==='target'?'目标话语：结合之前上下文，判断最后一条 seeker 的当前情绪':'对话级：通读全部文本，判断 seeker 的主要情绪';el('context').replaceChildren();for(const x of t.context){const p=document.createElement('pre');p.className=x.role;p.textContent=x.role+': '+x.text;el('context').append(p);}for(const f of [...Object.keys(options),'note'])el(f).value=a[f]||'';}
el('next').onclick=()=>{persist();const next=data.tasks[Math.min(pos+1,data.tasks.length-1)];if(next.scope==='dialogue'&&data.tasks.some(t=>t.scope==='target'&&!complete(t))){alert('请先完成全部 target 标签。');return;}pos=Math.min(pos+1,data.tasks.length-1);show();scrollTo(0,0);};el('prev').onclick=()=>{persist();pos=Math.max(0,pos-1);show();scrollTo(0,0);};
for(const f of [...Object.keys(options),'note','rater'])el(f).onchange=persist;
el('download').onclick=()=>{persist();if(!el('rater').value.trim()){alert('请填写匿名标注者 ID');return;}const result={package_id:data.package_id,rater_id:el('rater').value.trim(),annotations:Object.values(answers)};const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='E1_'+el('rater').value.replace(/[^a-zA-Z0-9_-]/g,'_')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
el('restore').onchange=async e=>{try{const r=JSON.parse(await e.target.files[0].text());if(r.package_id!==data.package_id)throw Error('标注包不匹配');if(!confirm('用该文件替换此浏览器已有进度？'))return;answers=Object.fromEntries(r.annotations.map(x=>[x.task_id,x]));el('rater').value=r.rater_id;localStorage.setItem(key,JSON.stringify(r));pos=0;show();}catch(err){alert(err.message);}};show();
</script>'''
    path.write_text(page.replace('OPTIONS', json.dumps(options)).replace('PAYLOAD', embedded), encoding='utf-8')


def prepare(args):
    root, e0, out = args.root.resolve(), args.e0.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError('Immutable output: choose a new directory')
    sys.path.insert(0, str(root))
    from psymm_generation.core import INPUTS, EMOTIONS, normalized_dialogue, digest as record_digest
    e0_summary = read(e0/'audit_summary.json')
    assignments = [json.loads(s) for s in (e0/'revision_split_assignments.jsonl').read_text().splitlines()]
    frame = select_frame(assignments, args.seed)
    needed = {(r['source'], r['source_index']): r for r in frame}
    input_evidence, cache, policy = {}, {}, Counter()
    esconv = []
    for source, relative in INPUTS.items():
        path = root/relative
        actual = file_hash(path)
        if actual != e0_summary['sources'][source]['sha256']:
            raise ValueError('Input changed since E0: ' + source)
        input_evidence[source] = dict(path=str(path), sha256=actual)
        records = read(path)
        for index, record in enumerate(records):
            if source == 'esconv':
                gold, pred = record.get('emotion_type'), record.get('predicted_emotion')
                if isinstance(gold, str) and isinstance(pred, str) and gold.strip() and pred.strip():
                    esconv.append((canonical(gold), canonical(pred), 1.0))
            key = source, index
            if key not in needed:
                continue
            row = needed[key]
            if record_digest(record) != row['record_sha256']:
                raise ValueError('E0 record digest mismatch')
            dialogues = list(normalized_dialogue(source, record, index))
            if len(dialogues) != 1:
                raise ValueError('Expected first-answer normalization')
            dialogue = dialogues[0]
            eligible = [i for i,t in enumerate(dialogue['turns']) if t['role'] == 'seeker'
                        and t['emotion_origin'] in ('dialogue_annotation','turn_annotation')]
            if not eligible:
                raise ValueError('Representative without annotated seeker')
            target = random.Random(digest([args.seed, source, index, 'target'])).choice(eligible)
            t = dialogue['turns'][target]
            row.update(eligible_turns=len(eligible), target_position=target,
                       raw_emotion=t['raw_emotion'],
                       predicted_fine=canonical(t['raw_emotion']), predicted_core=t['emotion'],
                       emotion_origin=t['emotion_origin'], language_heuristic=t['language'],
                       label_provenance=('source_ESConv_emotion_type_not_predicted_emotion'
                                         if source == 'esconv' else 'migrated_emotion_type_model_unverified'),
                       sample_id=digest([args.seed, source, index, row['group_id']])[:24])
            if source == 'esconv':
                row['stored_auto_fine'] = canonical(record['predicted_emotion'])
                row['stored_auto_core'] = FINE_MAP[row['stored_auto_fine']]

            if FINE_MAP.get(row['predicted_fine']) != t['emotion']:
                raise ValueError('Fine taxonomy conflicts with generator mapping')
            cache[key] = dialogue
            for turn in dialogue['turns']:
                policy[turn['emotion_origin']] += 1
    for raw, mapped in EMOTIONS.items():
        if FINE_MAP.get(canonical(raw)) != mapped:
            raise ValueError('Generator taxonomy changed: ' + raw)
    sample, coverage = select_samples(frame, args.seed, args.per_source, args.tail_min)
    tasks = public_tasks(sample, cache)
    package_id = digest(tasks)
    out.mkdir(parents=True)
    private = out/'administrator'; private.mkdir()
    packages = out/'annotators'; packages.mkdir()
    write(private/'sample_key.json', sample)
    write(private/'task_registry.json', dict(package_id=package_id, task_ids=[t['task_id'] for t in tasks]))
    write(private/'public_tasks.json', tasks)
    for name in ('A','B'):
        ordered = []
        for scope in ('target', 'dialogue'):
            part = [t for t in tasks if t['scope'] == scope]
            random.Random(digest([args.seed, name, scope])).shuffle(part)
            ordered.extend(part)
        render_html(packages/('annotator_' + name + '.html'), ordered, package_id)
    write(private/'adjudicator_template.json', dict(package_id=package_id, rater_id='', annotations=[]))
    fine_labels = sorted(FINE_MAP)
    raw_labels = sorted({x for g,p,w in esconv for x in (g,p)})
    exact = metrics(esconv, raw_labels)
    mapped = metrics([(FINE_MAP[g], FINE_MAP[p], w) for g,p,w in esconv], CORE)
    legacy = dict(status='COMPUTED_EXISTING_SOURCE_LABELS_NOT_NEW_HUMAN_VALIDATION',
        gold_field='emotion_type', prediction_field='predicted_emotion',
        model_identity='unverified', exact_fine=exact, mapped_core=mapped,
        source_sha256=input_evidence['esconv']['sha256'],
        caveat='Different file/fields from groundtruth.py; does not reproduce 71.3% or establish GPT-5.5 provenance.')
    write(private/'esconv_existing_pairs.json', legacy)
    compression = {}
    for source in INPUTS:
        rows = [r for r in frame if r['source'] == source]
        distribution = Counter(r['predicted_fine'] for r in rows)
        entropy = lambda counts: -sum((v/sum(counts.values())) * math.log2(v/sum(counts.values()))
                                     for v in counts.values() if v)
        mapped_counts = Counter()
        for fine, count in distribution.items():
            mapped_counts[FINE_MAP[fine]] += count
        compression[source] = dict(fine_counts=dict(distribution), core_counts=dict(mapped_counts),
            fine_entropy_bits=entropy(distribution), core_entropy_bits=entropy(mapped_counts),
            discarded_label_entropy_bits=entropy(distribution)-entropy(mapped_counts),
            interpretation='Deterministic category compression, not a measurement of perceptual information loss')
    write(private/'mapping_compression.json',compression)

    with (private/'representative_frame.jsonl').open('w', encoding='utf-8') as f:
        for row in frame:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest = dict(status='AWAITING_HUMAN_LABELS', seed=args.seed,
        per_source=args.per_source, tail_min=args.tail_min, sources=input_evidence,
        e0_assignments_sha256=file_hash(e0/'revision_split_assignments.jsonl'),
        normalization_sha256=file_hash(root/'psymm_generation/core.py'),
        script_sha256=file_hash(Path(__file__)), git_commit=subprocess.check_output(
            ['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        package_id=package_id, valid_records=len(assignments), representative_groups=len(frame),
        sample_dialogues=len(sample), tasks_per_rater=len(tasks), source_coverage=coverage,
        label_origin_counts_representative_turns=dict(policy), core_classes=list(CORE),
        fine_classes=fine_labels, aliases=ALIASES, fine_to_core=FINE_MAP,
        sample_unit='one hash-fixed representative record per global E0 group; one random eligible seeker target',
        estimands='natural: equal-source representative-group mean and representative eligible-turn ratio mean; tail: diagnostic only',
        source_language_hint={'esconv':'en','extes':'en','openr1':'en','psych8k':'en','psydial':'zh','psyqa':'zh'},
        historical_scope='Current E0-frozen inputs, not proof of historical paper-version pipeline.')
    write(out/'manifest.json', manifest)
    write(out/'artifact_hashes.json', {str(p.relative_to(out)):file_hash(p)
                                     for p in sorted(out.rglob('*')) if p.is_file()})
    print(json.dumps(dict(status=manifest['status'], sample_dialogues=len(sample),
                         tasks_per_rater=len(tasks), coverage=coverage,
                         esconv_fine_accuracy=exact['accuracy'], esconv_core_accuracy=mapped['accuracy']),
                     ensure_ascii=False))


def load_labels(path, registry, complete=True):
    data = read(path)
    if data.get('package_id') != registry['package_id'] or not str(data.get('rater_id','')).strip():
        raise ValueError('Wrong package or missing rater ID')
    rows = {}
    expected = set(registry['task_ids'])
    for row in data.get('annotations', []):
        tid = row.get('task_id')
        if tid not in expected or tid in rows:
            raise ValueError('Unknown or duplicate task ID')
        for field, allowed in [('fine', set(FINE_MAP)|set(SPECIAL)),
                               ('core', set(CORE)|set(SPECIAL)), ('language',set(LANGUAGES))]:
            if row.get(field) not in allowed:
                raise ValueError('Incomplete/invalid human annotation: ' + str(tid))
        rows[tid] = row
    if complete and set(rows) != expected:
        raise ValueError(f'AWAITING_HUMAN_LABELS: {len(rows)}/{len(expected)}')
    return data['rater_id'].strip(), rows


def score(args):
    bundle, out = args.bundle, args.output
    if out.exists():
        raise FileExistsError('Immutable output: choose a new directory')

    for name, sha in read(bundle/'artifact_hashes.json').items():
        if file_hash(bundle/name) != sha:
            raise ValueError('Bundle changed: '+name)
    registry = read(bundle/'administrator/task_registry.json')
    aid, a = load_labels(args.rater_a, registry)
    bid, b = load_labels(args.rater_b, registry)
    if aid == bid:
        raise ValueError('Two distinct independent annotator IDs required')
    disagreements = [tid for tid in registry['task_ids'] if any(a[tid][f] != b[tid][f] for f in FIELDS)]
    agreement = {}
    for scope in ('target','dialogue'):
        ids = [t for t in registry['task_ids'] if t.endswith(':'+scope)]
        agreement[scope] = {f:kappa([a[t][f] for t in ids],[b[t][f] for t in ids]) for f in FIELDS}
    out.mkdir(parents=True)
    write(out/'inter_annotator_agreement.json', agreement)
    if disagreements and args.adjudicated is None:

        write(out/'adjudication_required.json', dict(package_id=registry['package_id'], rater_id='',
            annotations=[dict(task_id=t, fine='', core='', language='', note='') for t in disagreements]))
        write(out/'status.json',dict(status='AWAITING_ADJUDICATION', disagreements=len(disagreements)))
        public = read(bundle/'administrator/public_tasks.json')
        adjudicator_tasks = [t for scope in ('target','dialogue') for t in public
                            if t['task_id'] in disagreements and t['scope'] == scope]
        render_html(out/'adjudicator.html',adjudicator_tasks,registry['package_id'])
        print('AWAITING_ADJUDICATION', len(disagreements)); return
    c = {}
    if args.adjudicated is not None:
        cid,c = load_labels(args.adjudicated,registry,complete=False)
        if cid in (aid,bid):
            raise ValueError('Independent third adjudicator required')
        if set(c) != set(disagreements):
            raise ValueError('Adjudication must cover exactly every disagreement')
    final = {t:(c[t] if t in c else a[t]) for t in registry['task_ids']}
    sample = read(bundle/'administrator/sample_key.json')
    reports = {}
    for cohort in ('natural','tail'):
        part = [r for r in sample if r['cohort'] == cohort]
        strata = {'all':part}
        for source in sorted({r['source'] for r in part}):
            strata['source:'+source] = [r for r in part if r['source'] == source]
        for lang in LANGUAGES:
            for scope in ('target','dialogue'):
                strata['language:'+lang+':'+scope] = [r for r in part if
                    final[r['sample_id']+':'+scope]['language'] == lang]
        for stratum, rows in strata.items():
            for scope in ('target','dialogue'):
                if stratum.startswith('language:') and not stratum.endswith(':'+scope):
                    continue
                for weighted in (False,True) if cohort == 'natural' else (False,):
                    weight_field = 'turn_weight' if scope == 'target' else 'group_weight'
                    fine_pairs, core_pairs, mapping_pairs = [], [], []
                    historical_auto_fine, historical_auto_core = [], []
                    uncertain, core_unsupported = 0, 0
                    rescued, concealed = 0, 0
                    for row in rows:
                        gold = final[row['sample_id']+':'+scope]
                        w = row[weight_field] if weighted else 1.0
                        if gold['fine'] != 'uncertain':
                            fine_pairs.append((gold['fine'],row['predicted_fine'],w))
                            if 'stored_auto_fine' in row:
                                historical_auto_fine.append((gold['fine'],row['stored_auto_fine'],w))
                        if gold['core'] != 'uncertain':
                            core_pairs.append((gold['core'],row['predicted_core'],w))
                            if gold['core'] not in CORE:
                                core_unsupported += 1
                            if 'stored_auto_core' in row:
                                historical_auto_core.append((gold['core'],row['stored_auto_core'],w))
                        else:
                            uncertain += 1
                        if gold['fine'] in FINE_MAP and gold['core'] in CORE:
                            mapping_pairs.append((gold['core'],FINE_MAP[gold['fine']],w))
                            fine_wrong = gold['fine'] != row['predicted_fine']
                            mapped_same = FINE_MAP[gold['fine']] == row['predicted_core']
                            concealed += int(fine_wrong and mapped_same)
                            rescued += int(FINE_MAP[gold['fine']] != gold['core'])
                    key = f'{cohort}/{stratum}/{scope}/' + ('population_ratio' if weighted else 'sample')
                    reports[key] = dict(n=len(rows), uncertain_core=uncertain,
                        other_core=core_unsupported,
                        core_coverage=(len(core_pairs)/len(rows) if rows else None),
                        fine=metrics(fine_pairs,sorted(FINE_MAP)+['other']),
                        core=metrics(core_pairs,list(CORE)+['other']),
                        human_fine_mapping_vs_direct_core=metrics(mapping_pairs,CORE),
                        fine_mismatch_concealed_by_mapping_count=concealed,
                        human_fine_to_core_disagreement_count=rescued)
                    if historical_auto_fine or historical_auto_core:
                        reports[key]['esconv_stored_prediction_vs_manual'] = dict(
                            fine=metrics(historical_auto_fine,sorted(FINE_MAP)+['other']),
                            core=metrics(historical_auto_core,list(CORE)+['other']))
    write(out/'results.json',dict(status='COMPLETE_TEXT_VALIDATION', metrics=reports,
        primary_comparison='Labels actually used by current generator vs new adjudicated human labels; pooled result is not an automatic-model accuracy',
        agreement=agreement, disagreements=len(disagreements),
        rater_ids=[aid,bid], adjudicator_id=cid if args.adjudicated else None,
        source_bundle_sha256=file_hash(bundle/'manifest.json'),
        label_hashes={'a':file_hash(args.rater_a),'b':file_hash(args.rater_b),
                      'adjudicated':file_hash(args.adjudicated) if args.adjudicated else None},
        limitations=['No synthesized-media validation (E2 pending)',
            'No bootstrap CI in this report; descriptive estimates only',
            'Uncertain excluded from label accuracy; coverage separately reported',
            'Fine scores for seven-class sources do not validate unavailable finer automatic labels']))
    print('COMPLETE_TEXT_VALIDATION',len(sample))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command',required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--root',type=Path,required=True)
    a.add_argument('--e0',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--seed',type=int,default=20261019)
    a.add_argument('--per-source',type=int,default=100)
    a.add_argument('--tail-min',type=int,default=8)
    b = sub.add_parser('score')
    b.add_argument('--bundle',type=Path,required=True)
    b.add_argument('--rater-a',type=Path,required=True)
    b.add_argument('--rater-b',type=Path,required=True)
    b.add_argument('--adjudicated',type=Path)
    b.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    (prepare if args.command=='prepare' else score)(args)


if __name__=='__main__':
    main()
