#!/usr/bin/env python3
"""Read-only E2 media coverage and reference-identity metadata audit.

The script reads mapping JSON/SQLite metadata and checks file existence only.
It does not open, decode, or score audio/video content.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "esconv": {
        "folder": "infer_outputs_esconv",
        "maps": ["audio_reference_map_emo.json", "audio_reference_map.json"],
        "audio_dirs": ["Openvoice_v2_outputs", "OpenVoiceEMO_outputs", "OpenVoice_outputs"],
        "video_dirs": ["DICE_Talk_outputs"],
    },
    "extes": {
        "folder": "infer_outputs_extes",
        "maps": ["audio_reference_map_emo.json", "audio_reference_map.json"],
        "audio_dirs": ["OpenVoiceEMO_outputs", "OpenVoice_outputs"],
        "video_dirs": ["DICE_Talk_outputs"],
        "state_db": "direct_generation/direct_state.sqlite",
    },
    "openr1_psy": {
        "folder": "infer_outputs_openr1-psy",
        "maps": ["audio_reference_map_emo.json", "audio_reference_map.json"],
        "audio_dirs": ["OpenVoiceEMO_outputs", "OpenVoice_outputs"],
        "video_dirs": ["DICE_Talk_outputs"],
    },
    "psydial": {
        "folder": "infer_outputs_psydial",
        "maps": ["audio_reference_map_emo.json", "audio_reference_map.json"],
        "audio_dirs": ["OpenVoiceEMO_outputs", "OpenVoice_outputs"],
        "video_dirs": ["DICE_Talk_outputs"],
    },
    "psyqa": {
        "folder": "infer_outputs_psyqa",
        "maps": ["audio_reference_map_emo.json", "audio_reference_map.json"],
        "audio_dirs": ["OpenVoiceEMO_outputs", "OpenVoice_outputs"],
        "video_dirs": ["DICE_Talk_outputs"],
        "state_db": "direct_generation/direct_state.sqlite",
    },
    "psy8k": {
        "folder": "infer_outputs_psy8k",
        "maps": ["audio_reference_map_emo.json", "audio_reference_map.json"],
        "audio_dirs": ["OpenVoiceEMO_outputs", "OpenVoice_outputs"],
        "video_dirs": ["DICE_Talk_outputs"],
    },
}

PROFILE_RE = re.compile(r"_(\d+)(?:\.[^.]+)?$")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def identity_from_reference(value: Any) -> str | None:
    if not value:
        return None
    match = PROFILE_RE.search(Path(str(value)).name)
    return match.group(1) if match else None


def first_value(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def base_record(dataset: str, pipeline: str, sample_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    reference = first_value(raw, "reference_path", "reference_audio", "speaker_ref", "listener_ref", "reference")
    reference_obj = reference if isinstance(reference, dict) else {}
    if isinstance(reference, dict):
        reference_path = first_value(reference, "audio", "segment", "path")
        identity = reference.get("identity") or identity_from_reference(reference_path)
        reference_clip = reference.get("segment") or Path(str(reference_path)).name if reference_path else None
    else:
        reference_path = reference
        identity = identity_from_reference(reference)
        reference_clip = Path(str(reference)).name if reference else None
    dialogue = first_value(raw, "dialogue_id", "dialogue_scope", "conversation", "post_id", "dialog_id")
    turn = first_value(raw, "turn_index", "turn", "turn_index", "round", "utt_id")
    emotion = first_value(raw, "intended_emotion", "emotion", "emotion_label", "target_emotion", "raw_emotion")
    language = first_value(raw, "language", "lang")
    role = first_value(raw, "role", "speaker_role", "original_role")
    return {
        "dataset": dataset,
        "pipeline": pipeline,
        "sample_id": sample_id,
        "dialogue_id": str(dialogue) if dialogue is not None else "unknown",
        "turn": str(turn) if turn is not None else "unknown",
        "role": str(role).lower() if role is not None else "unknown",
        "emotion": str(emotion).lower() if emotion is not None else "unknown",
        "language": str(language).upper() if language is not None else "unknown",
        "utterance_text": str(first_value(raw, "text", "original_text", "utterance", "target_text") or ""),
        "identity_id": str(identity) if identity is not None else None,
        "reference_clip": str(reference_clip) if reference_clip else None,
        "reference_raw": str(reference_path) if reference_path else "",
    }


def find_dir(folder: Path, names: list[str]) -> Path | None:
    return next((folder / name for name in names if (folder / name).is_dir()), None)


def video_index(video_dir: Path | None) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    if video_dir is None:
        return index
    for path in video_dir.iterdir():
        if path.is_file() and path.suffix.lower() == ".mp4":
            index[path.stem].append(path)
            match = re.fullmatch(r"(.+)_S\d+", path.stem)
            if match:
                index[match.group(1)].append(path)
    return index


def locate_video(dataset: str, sample_id: str, raw: dict[str, Any], index: dict[str, list[Path]]) -> tuple[Path | None, str]:
    stem = Path(sample_id).stem



    base = stem
    matches = list(dict.fromkeys(index.get(base, [])))
    if len(matches) == 1:
        return matches[0], "candidate_found"
    if len(matches) > 1:
        return None, "ambiguous_candidates"
    return None, "candidate_not_found"


def load_map_stream(root: Path, dataset: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = DATASETS[dataset]
    folder = root / spec["folder"]
    map_path = next((folder / name for name in spec["maps"] if (folder / name).is_file()), None)
    if map_path is None:
        return {"dataset": dataset, "pipeline": "legacy_map", "status": "NO_MAP"}, []
    audio_dir = find_dir(folder, spec["audio_dirs"])
    video_dir = find_dir(folder, spec["video_dirs"])
    videos = video_index(video_dir)
    if audio_dir is None:
        return {
            "dataset": dataset, "pipeline": "legacy_map", "status": "MAP_WITHOUT_AUDIO_OUTPUT_DIR",
            "map": str(map_path), "mapped_samples": 0,
        }, []
    raw_map = read_json(map_path)
    if not isinstance(raw_map, dict):
        return {"dataset": dataset, "pipeline": "legacy_map", "status": "UNSUPPORTED_MAP", "map": str(map_path)}, []

    records = []
    for key, value in raw_map.items():
        raw = value if isinstance(value, dict) else {}
        row = base_record(dataset, "legacy_map", Path(key).name, raw)
        if row["dialogue_id"] == "unknown":
            row["dialogue_id"] = Path(key).stem
        audio_path = audio_dir / row["sample_id"]
        video_path, video_match_status = locate_video(dataset, row["sample_id"], raw, videos) if video_dir else (None, "no_video_output_dir")
        video_exists = bool(video_path and video_path.is_file())
        row.update({
            "frame_status": "mapped",
            "audio_expected": True,
            "audio_status": "present" if audio_path.is_file() else "missing_from_map",
            "audio_exists": audio_path.is_file(),
            "audio_path": str(audio_path),
            "video_expected": video_dir is not None,
            "video_status": ("present" if video_exists else video_match_status) if video_dir else "no_video_output_dir",
            "video_exists": video_exists,
            "video_path": str(video_path) if video_path else "",
            "generation_error": "",
        })
        records.append(row)
    summary = summarize_stream(dataset, "legacy_map", records)
    summary.update({
        "status": "MAPPED",
        "map": str(map_path),
        "audio_dir": str(audio_dir),
        "video_dir": str(video_dir) if video_dir else None,
        "video_expected_count": sum(r["video_expected"] for r in records),
        "video_directory_absent": video_dir is None,
        "warning": "A missing video directory means this pipeline has no observed video output location; it is not counted as per-item video failure.",
    })
    return summary, records


def load_direct_stream(root: Path, dataset: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = DATASETS[dataset]
    if "state_db" not in spec:
        return {"dataset": dataset, "pipeline": "direct_generation", "status": "NOT_APPLICABLE"}, []
    folder = root / spec["folder"]
    db_path = folder / spec["state_db"]
    if not db_path.is_file():
        return {"dataset": dataset, "pipeline": "direct_generation", "status": "NO_STATE_DB"}, []

    connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("BEGIN")
    items = connection.execute("SELECT asset_id, item FROM items").fetchall()
    media = connection.execute("SELECT asset_id, stage, status, details FROM media").fetchall()
    skipped = connection.execute("SELECT source_index, reason FROM skipped").fetchall()
    connection.close()
    status_by_asset = {(r["asset_id"], r["stage"]): (r["status"], r["details"]) for r in media}
    records = []
    for item_row in items:
        item = json.loads(item_row["item"])
        asset_id = item_row["asset_id"]
        row = base_record(dataset, "direct_generation", asset_id, item)
        row["dialogue_id"] = str(item.get("dialogue_id") or row["dialogue_id"])
        row["turn"] = str(item.get("turn_index", row["turn"]))
        state = status_by_asset.get((asset_id, "audio"))
        audio_status = state[0] if state else "not_recorded"
        audio_path = folder / "direct_generation" / "audio" / f"{asset_id}.wav"
        audio_exists = audio_path.is_file()
        if state and state[1]:
            details = json.loads(state[1])
            candidate = details.get("output_audio") or details.get("audio")
            if candidate:
                audio_path = Path(candidate)
                audio_exists = audio_path.is_file()
        row.update({
            "frame_status": "generation_plan",
            "asset_id": asset_id,
            "source_index": item.get("source_index"),
            "audio_expected": True,
            "audio_status": audio_status,
            "audio_exists": audio_exists,
            "audio_path": str(audio_path),
            "video_expected": False,
            "video_status": "not_requested_by_audio_pipeline",
            "video_exists": False,
            "video_path": "",
            "generation_error": "",
            "state_details": state[1] if state else "",
        })
        if audio_status == "failed":
            try:
                row["generation_error"] = json.loads(state[1] or "{}").get("error", "")
            except (TypeError, json.JSONDecodeError):
                row["generation_error"] = "failure_details_unavailable"
        records.append(row)
    summary = summarize_stream(dataset, "direct_generation", records)
    summary.update({
        "status": "STATE_DB_READ",
        "state_db": str(db_path),
        "state_stage_status_counts": dict(Counter(f"{key[1]}:{value[0]}" for key, value in status_by_asset.items())),
        "items_without_audio_state": sum((r["asset_id"], "audio") not in status_by_asset for r in records),
        "skipped_input_count": len(skipped),
        "skipped_reason_counts": dict(Counter(r["reason"] for r in skipped)),
        "video_expected_count": 0,
        "warning": "This state DB covers its direct-generation input plan only. It records audio generation; absent video is not a failure for this audio-only pipeline.",
    })
    return summary, records


def summarize_stream(dataset: str, pipeline: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    ids = Counter(r["identity_id"] for r in records if r["identity_id"] is not None)
    return {
        "dataset": dataset,
        "pipeline": pipeline,
        "mapped_samples": len(records),
        "audio_present": sum(r["audio_exists"] for r in records),
        "audio_missing_or_unmaterialized": sum(not r["audio_exists"] for r in records),
        "audio_state_counts": dict(Counter(r["audio_status"] for r in records)),
        "audio_state_file_counts": dict(Counter(
            f"{r['audio_status']}:{'present' if r['audio_exists'] else 'missing'}" for r in records
        )),
        "video_present": sum(r["video_exists"] for r in records),
        "video_missing_where_expected": sum(r["video_expected"] and not r["video_exists"] for r in records),
        "identity_missing": sum(r["identity_id"] is None for r in records),
        "language_unknown": sum(r["language"] == "unknown" for r in records),
        "identity_audit": identity_summary(records),
    }


def identity_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_dialogue_role: dict[tuple[str, str], set[str]] = defaultdict(set)
    roles_by_dialogue: dict[str, set[str]] = defaultdict(set)
    identities_by_dialogue: dict[str, set[str]] = defaultdict(set)
    freq: Counter[str] = Counter()
    clips: Counter[str] = Counter()
    emotion: Counter[tuple[str, str]] = Counter()
    language: Counter[tuple[str, str]] = Counter()
    for row in records:
        identity = row["identity_id"]
        if identity is None:
            continue
        freq[identity] += 1
        clips[row["reference_clip"] or "unknown"] += 1
        by_dialogue_role[(row["dialogue_id"], row["role"])].add(identity)
        if row["role"] != "unknown":
            roles_by_dialogue[row["dialogue_id"]].add(row["role"])
            identities_by_dialogue[row["dialogue_id"]].add(identity)
        emotion[(identity, row["emotion"])] += 1
        language[(identity, row["language"])] += 1
    return {
        "unique_reference_identity_ids": len(freq),
        "unique_reference_clips": len(clips),
        "dialogue_role_groups_with_identity_drift": sum(len(values) > 1 for values in by_dialogue_role.values()),
        "dialogues_with_same_identity_across_roles": sum(
            len(roles_by_dialogue[key]) >= 2 and len(identities_by_dialogue[key]) < 2
            for key in roles_by_dialogue
        ),
        "identity_occurrence_top20": freq.most_common(20),
        "reused_reference_clips_top20": [(name, count) for name, count in clips.most_common(20) if count > 1],
        "identity_by_emotion": [
            {"identity_id": identity, "emotion": label, "count": count}
            for (identity, label), count in sorted(emotion.items())
        ],
        "identity_by_language": [
            {"identity_id": identity, "language": label, "count": count}
            for (identity, label), count in sorted(language.items())
        ],
    }


def stratified_sample(records: list[dict[str, Any]], budget: int, seed: int, require_av: bool) -> list[dict[str, Any]]:
    eligible = [r for r in records if r["audio_exists"] and (not require_av or r["video_exists"])]
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        groups[(row["dataset"], row["pipeline"], row["language"], row["emotion"])].append(row)
    rng = random.Random(seed)
    for rows in groups.values():
        rng.shuffle(rows)
    datasets = sorted({key[0] for key in groups})
    cursors = {dataset: 0 for dataset in datasets}
    selected: list[dict[str, Any]] = []
    while len(selected) < min(budget, len(eligible)):
        progress = False
        for dataset in datasets:
            keys = [key for key in sorted(groups) if key[0] == dataset and groups[key]]
            if keys and len(selected) < budget:
                key = keys[cursors[dataset] % len(keys)]
                cursors[dataset] += 1
                selected.append(groups[key].pop())
                progress = True
        if not progress:
            break
    return selected


def identity_panels(records: list[dict[str, Any]], budget: int, seed: int) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["audio_exists"] and row["video_exists"] and row["identity_id"] and row["role"] != "unknown":
            groups[(row["dataset"], row["pipeline"], row["dialogue_id"], row["role"])].append(row)
    candidates = [sorted(rows, key=lambda row: (row["turn"], row["sample_id"])) for rows in groups.values() if len(rows) >= 2]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    by_dataset: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for panel in candidates:
        by_dataset[panel[0]["dataset"]].append(panel)
    panels: list[list[dict[str, Any]]] = []
    datasets = sorted(by_dataset)
    cursors = {name: 0 for name in datasets}
    while len(panels) < min(budget, len(candidates)):
        advanced = False
        for name in datasets:
            if cursors[name] < len(by_dataset[name]) and len(panels) < budget:
                panels.append(by_dataset[name][cursors[name]][:4])
                cursors[name] += 1
                advanced = True
        if not advanced:
            break
    return panels


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def review_link(output: Path, modality: str, review_id: str, source: str) -> Path:
    folder = output / "blind_media" / modality
    folder.mkdir(parents=True, exist_ok=True)
    suffix = Path(source).suffix.lower()
    link = folder / f"{review_id}{suffix}"
    target = Path(source).resolve()
    if link.is_symlink():
        if link.resolve() != target:
            link.unlink()
            link.symlink_to(target)
    elif link.exists():
        raise FileExistsError(f"review path already exists and is not a symlink: {link}")
    else:
        link.symlink_to(target)
    return link


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--review-budget", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    if args.review_budget < 0:
        parser.error("--review-budget must be non-negative")
    root, output = args.project_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for loader in (load_map_stream, load_direct_stream):
            summary, rows = loader(root, dataset)
            summaries.append(summary)
            records.extend(rows)

    audio_sample = stratified_sample(records, args.review_budget, args.seed, require_av=False)
    av_sample = stratified_sample(records, args.review_budget, args.seed + 1, require_av=True)
    identity_sample = identity_panels(records, min(60, args.review_budget), args.seed + 4)
    random.Random(args.seed + 2).shuffle(audio_sample)
    random.Random(args.seed + 3).shuffle(av_sample)
    missing = [r for r in records if (r["audio_expected"] and not r["audio_exists"]) or (r["video_expected"] and not r["video_exists"])]
    report = {
        "experiment": "E2 media coverage and identity metadata audit",
        "verification_status": "METADATA_AUDIT_ONLY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "seed": args.seed,
        "review_budget_per_pack": args.review_budget,
        "streams": summaries,
        "frame_record_count": len(records),
        "audio_review_sample_count": len(audio_sample),
        "complete_av_review_sample_count": len(av_sample),
        "identity_panel_count": len(identity_sample),
        "expected_media_missing_count": len(missing),
        "limitations": [
            "Only JSON/SQLite metadata and filesystem existence were inspected; no audio/video content was decoded or scored.",
            "An absent output directory is reported at stream level and is not converted into per-item generation failures.",
            "Legacy maps do not preserve attempt-level failure causes; only direct-generation SQLite states can distinguish recorded failures from unrecorded items.",
            "Direct-generation state is read in a SQLite transaction, but file existence is checked separately; active generation can cause transient state/file mismatches.",
            "Identity IDs come from explicit direct-generation reference metadata or the numeric suffix of AvaMERG reference filenames; validate identity semantics against source metadata.",
            "Legacy map language is unknown when it is not recorded; no language is inferred from dataset names.",
            "Human perceptual ratings and objective audio/video quality measures remain outstanding; these templates are sampling aids only.",
        ],
    }
    (output / "e2_metadata_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["dataset", "pipeline", "sample_id", "asset_id", "source_index", "dialogue_id", "turn", "role",
              "emotion", "language", "utterance_text", "identity_id", "reference_clip", "audio_expected", "audio_status",
              "audio_exists", "audio_path", "video_expected", "video_status", "video_exists", "video_path", "generation_error"]
    write_csv(output / "e2_sample_frame.csv", records, fields)
    write_csv(output / "e2_missing_outputs.csv", missing, fields)
    review_key = []
    audio_rows = []
    for number, row in enumerate(audio_sample, start=1):
        review_id = f"E2A-{number:04d}"
        review_key.append({"review_id": review_id, "review_type": "audio", **row})
        audio_rows.append({
            "review_id": review_id,
            "audio_path": str(review_link(output, "audio", review_id, row["audio_path"])),
            "utterance_text": row["utterance_text"],
            "speech_naturalness_1_5": "", "speech_artifact": "", "prosody_issue": "",
            "text_audio_emotion_1_5": "", "notes": "",
        })
    write_csv(output / "e2_audio_review_template.csv", audio_rows, [
        "review_id", "audio_path", "utterance_text", "speech_naturalness_1_5", "speech_artifact", "prosody_issue",
        "text_audio_emotion_1_5", "notes",
    ])
    av_rows = []
    for number, row in enumerate(av_sample, start=1):
        review_id = f"E2V-{number:04d}"
        review_key.append({"review_id": review_id, "review_type": "audio_video", **row})
        av_rows.append({
            "review_id": review_id,
            "audio_path": str(review_link(output, "audio", review_id, row["audio_path"])),
            "video_path": str(review_link(output, "video", review_id, row["video_path"])),
            "utterance_text": row["utterance_text"],
            "speech_naturalness_1_5": "", "speech_artifact": "", "prosody_issue": "",
            "text_audio_emotion_1_5": "", "facial_emotion_1_5": "", "crossmodal_emotion_1_5": "",
            "lip_sync_1_5": "", "overall_av_quality_1_5": "", "notes": "",
        })
    write_csv(output / "e2_av_review_template.csv", av_rows, [
        "review_id", "audio_path", "video_path", "utterance_text", "speech_naturalness_1_5", "speech_artifact", "prosody_issue",
        "text_audio_emotion_1_5", "facial_emotion_1_5", "crossmodal_emotion_1_5", "lip_sync_1_5",
        "overall_av_quality_1_5", "notes",
    ])
    identity_rows = []
    for panel_number, panel in enumerate(identity_sample, start=1):
        panel_id = f"E2I-{panel_number:04d}"
        for clip_number, row in enumerate(panel, start=1):
            clip_id = f"{panel_id}-C{clip_number:02d}"
            review_key.append({"review_id": clip_id, "review_type": "identity_panel", **row})
            identity_rows.append({
                "panel_id": panel_id,
                "clip_id": clip_id,
                "video_path": str(review_link(output, "identity_video", clip_id, row["video_path"])),
                "identity_stability_1_5": "",
                "visible_identity_change": "",
                "notes": "",
            })
    write_csv(output / "e2_identity_panel_template.csv", identity_rows, [
        "panel_id", "clip_id", "video_path", "identity_stability_1_5", "visible_identity_change", "notes",
    ])
    write_csv(output / "e2_review_key.csv", review_key, ["review_id", "review_type"] + fields)
    print(json.dumps({
        "output": str(output), "frame_records": len(records), "audio_review_sample": len(audio_sample),
        "complete_av_review_sample": len(av_sample), "identity_panels": len(identity_sample),
        "identity_panel_clips": len(identity_rows), "expected_media_missing": len(missing),
        "status": "METADATA_AUDIT_ONLY",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
