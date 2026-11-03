#!/usr/bin/env python3
"""Run reproducible automatic E2 media checks on a frozen blind-review pack.

Outputs ASR/text-match metrics, video technical metadata, and face-embedding
similarity for identity panels. These proxies do not replace human ratings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def latest_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the latest attempt per task; raw JSONL retains every attempt."""
    latest = {}
    for row in rows:
        latest[(row.get("metric_type"), row.get("review_id"))] = row
    return list(latest.values())


def pending_tasks(tasks, records, retry_failed=False):
    done = {(r.get("metric_type"), r.get("review_id"))
            for r in latest_records(records) if not retry_failed or not r.get("error")}
    return [(kind, row) for kind, row in tasks
            if (kind, row.get("review_id", row.get("clip_id"))) not in done]


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_media(path: Path) -> dict[str, Any]:
    import av
    container = av.open(str(path))
    streams = []
    try:
        for stream in container.streams:
            row: dict[str, Any] = {"type": stream.type, "codec": stream.codec_context.name}
            if stream.duration is not None and stream.time_base is not None:
                row["duration_s"] = float(stream.duration * stream.time_base)
            if stream.type == "audio":
                row.update({"sample_rate": stream.codec_context.sample_rate,
                            "channels": stream.codec_context.channels})
            elif stream.type == "video":
                row.update({"width": stream.codec_context.width,
                            "height": stream.codec_context.height,
                            "average_rate": str(stream.average_rate) if stream.average_rate else None,
                            "frames": stream.frames})
            streams.append(row)
        duration = float(container.duration / av.time_base) if container.duration else None
    finally:
        container.close()
    return {"duration_s": duration, "streams": streams}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    return "".join(ch for ch in text if ch.isalnum())


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def text_match(reference: str, hypothesis: str, language: str) -> dict[str, Any]:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return {"reference_chars": 0, "cer": None, "wer": None}
    distance = edit_distance(ref, hyp)

    cer = distance / len(ref)
    wer = None
    if language and not language.lower().startswith("zh"):
        ref_words = normalize_words(reference)
        hyp_words = normalize_words(hypothesis)
        wer = edit_distance(ref_words, hyp_words) / max(len(ref_words), 1)
    return {"reference_chars": len(ref), "cer": cer, "wer": wer}


class Evaluator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.asr = None
        self.face_detector = None
        self.face_embedder = None

    def load_asr(self) -> None:
        from faster_whisper import WhisperModel
        self.asr = WhisperModel(
            str(self.args.asr_model), device=self.args.device, compute_type=self.args.compute_type,
            local_files_only=True,
        )

    def load_face_models(self) -> None:
        import torch
        from facenet_pytorch import InceptionResnetV1, MTCNN
        device = torch.device(self.args.device)
        self.face_detector = MTCNN(keep_all=True, device=device, post_process=True)
        self.face_embedder = InceptionResnetV1(classify=False, pretrained=None).eval().to(device)
        state = torch.load(self.args.facenet_weights, map_location="cpu", weights_only=True)
        state = {key: value for key, value in state.items() if not key.startswith("logits.")}
        self.face_embedder.load_state_dict(state, strict=True)

    def transcribe(self, row: dict[str, str]) -> dict[str, Any]:
        from faster_whisper import __version__ as fw_version
        audio_path = Path(row["audio_path"])
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        segments, info = self.asr.transcribe(
            str(audio_path), language=None, beam_size=5, vad_filter=False,
            condition_on_previous_text=False,
        )
        parts = []
        for segment in segments:
            parts.append({"start": segment.start, "end": segment.end, "text": segment.text})
        transcript = "".join(part["text"] for part in parts).strip()
        reference = row.get("utterance_text", "")
        match = text_match(reference, transcript, info.language)
        probe = probe_media(audio_path)
        return {
            "review_id": row["review_id"], "metric_type": "audio", "audio_path": str(audio_path),
            "dataset": row.get("dataset"), "pipeline": row.get("pipeline"),
            "audio_sha256": file_sha256(audio_path), "detected_language": info.language,
            "language_probability": info.language_probability, "duration_s": info.duration,
            "transcript": transcript, "segments": parts, **match,
            "audio_streams": [s for s in probe["streams"] if s["type"] == "audio"],
            "asr_version": fw_version,
        }

    def video_metrics(self, row: dict[str, str]) -> dict[str, Any]:
        import cv2
        facenet_version = importlib.metadata.version("facenet-pytorch")
        video_path = Path(row["video_path"])
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        probe = probe_media(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not decode video")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else None
        frames_checked = 0
        face_embeddings = []
        frame_indices = sorted(set([0, max(frame_count // 2, 0), max(frame_count - 1, 0)]))
        try:
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                frames_checked += 1
                if self.face_detector is not None:
                    from PIL import Image
                    import torch
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(rgb)
                    boxes, _ = self.face_detector.detect(image)
                    if boxes is None or len(boxes) == 0:
                        continue
                    areas = [(max(b[2] - b[0], 0) * max(b[3] - b[1], 0), i)
                             for i, b in enumerate(boxes) if b is not None and all(math.isfinite(float(x)) for x in b)]
                    if not areas:
                        continue
                    _, largest = max(areas)
                    faces = self.face_detector.extract(image, boxes[largest:largest + 1], save_path=None)
                    if faces is None:
                        continue
                    face = faces[:1]
                    with torch.no_grad():
                        embedding = self.face_embedder(face.to(self.args.device)).cpu().numpy()[0]
                    norm = float((embedding ** 2).sum() ** 0.5)
                    if norm:
                        face_embeddings.append((embedding / norm).tolist())
        finally:
            cap.release()
        stream_info = probe.get("streams", [])
        audio_stream = any(s.get("type") == "audio" for s in stream_info)
        return {
            "review_id": row.get("review_id", row.get("clip_id")),
            "panel_id": row.get("panel_id"), "clip_id": row.get("clip_id"),
            "dataset": row.get("dataset"), "pipeline": row.get("pipeline"),
            "metric_type": "video", "video_path": str(video_path),
            "video_sha256": file_sha256(video_path), "duration_s": duration,
            "fps": fps, "frame_count": frame_count, "width": width, "height": height,
            "frames_decoded_for_face_check": frames_checked, "audio_stream_present": audio_stream,
            "face_detected_frames": len(face_embeddings), "face_embeddings": face_embeddings,
            "probe": probe, "facenet_version": facenet_version,
        }


def summarize(rows: list[dict[str, Any]], out: Path) -> None:
    rows = latest_records(rows)
    successful = [r for r in rows if not r.get("error")]
    metrics: dict[str, list[float]] = defaultdict(list)
    for row in successful:
        for field in ("cer", "wer", "duration_s", "fps", "width", "height"):
            value = row.get(field)
            if isinstance(value, (int, float)) and math.isfinite(value):
                metrics[field].append(float(value))
    summary: dict[str, Any] = {
        "experiment": "E2 automatic media quality proxies",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(rows), "successful_record_count": len(successful),
        "failed_record_count": len(rows) - len(successful),
        "metric_means": {name: sum(values) / len(values) for name, values in metrics.items() if values},
        "failure_counts": {},
        "limitations": [
            "ASR CER/WER measures transcript agreement only where source text exists; it is not perceived speech naturalness.",
            "Three sampled frames and FaceNet cosine similarities are identity continuity proxies, not identity recognition accuracy.",
            "This script does not calculate SyncNet lip-sync scores or human perceptual ratings.",
            "Failure and missing-sample denominators must be interpreted with the frozen sampling frame.",
        ],
    }
    failures: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("error"):
            failures[row.get("metric_type", "unknown")] += 1
    summary["failure_counts"] = dict(failures)
    panels: dict[str, list[list[float]]] = defaultdict(list)
    for row in successful:
        panel_id = row.get("panel_id")
        embeddings = row.get("face_embeddings", [])
        if panel_id and embeddings:
            vector = [sum(frame[i] for frame in embeddings) / len(embeddings)
                      for i in range(len(embeddings[0]))]
            norm = math.sqrt(sum(value * value for value in vector))
            if norm:
                panels[panel_id].append([value / norm for value in vector])
    panel_scores = []
    for panel_id, vectors in sorted(panels.items()):
        similarities = []
        for i, left in enumerate(vectors):
            for right in vectors[i + 1:]:
                similarities.append(sum(a * b for a, b in zip(left, right)))
        if similarities:
            panel_scores.append({"panel_id": panel_id, "clips_with_faces": len(vectors),
                                 "pairwise_cosine_mean": sum(similarities) / len(similarities),
                                 "pairwise_cosine_min": min(similarities)})
    summary["identity_panel_face_similarity"] = panel_scores
    (out / "e2_auto_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_results_csv(out / "e2_auto_metrics.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asr-model", type=Path, required=True)
    parser.add_argument("--facenet-weights", type=Path, required=True)
    parser.add_argument("--compute-type", default="float16", choices=["float16", "int8_float16", "float32"])
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-face", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", help="Retry only failed or unfinished tasks; keep successful results")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="Optional small smoke-test limit")
    args = parser.parse_args()
    if not args.pack.is_dir():
        parser.error(f"blind-review pack not found: {args.pack}")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "e2_auto_metrics.jsonl"
    tasks: list[tuple[str, dict[str, str]]] = []
    if not args.skip_asr:
        audio_rows = read_csv(args.pack / "e2_audio_review_template.csv")
        tasks.extend(("audio", row) for row in audio_rows)
    if not args.skip_face:
        panel_rows = read_csv(args.pack / "e2_identity_panel_template.csv")
        tasks.extend(("video", row) for row in panel_rows)
        av_rows = read_csv(args.pack / "e2_av_review_template.csv")
        tasks.extend(("video", row) for row in av_rows)
    key_path = args.pack / "e2_review_key.csv"
    key = {r["review_id"]: r for r in read_csv(key_path)} if key_path.exists() else {}
    for _, row in tasks:
        metadata = key.get(row.get("review_id", row.get("clip_id")), {})
        for field in ("dataset", "pipeline", "language", "dialogue_id", "role", "emotion"):
            row[field] = metadata.get(field, row.get(field, ""))
    tasks = pending_tasks(tasks, read_jsonl(checkpoint), args.retry_failed)
    if args.limit:
        tasks = tasks[:args.limit]
    evaluator = Evaluator(args)
    if any(kind == "audio" for kind, _ in tasks):
        evaluator.load_asr()
    if any(kind == "video" for kind, _ in tasks):
        evaluator.load_face_models()

    count = 0
    for metric_type, row in tasks:
        review_id = row["review_id"] if metric_type != "video" or "review_id" in row else row.get("clip_id", "")
        try:
            if metric_type == "audio":
                result = evaluator.transcribe(row)
            else:
                result = evaluator.video_metrics(row)
                if "review_id" not in result:
                    result["review_id"] = row.get("clip_id", review_id)
            result["metric_type"] = metric_type
        except Exception as exc:
            result = {"review_id": review_id, "metric_type": metric_type,
                      "source_id": row.get("clip_id", review_id),
                      "path": row.get("audio_path", row.get("video_path", "")),
                      "error": f"{type(exc).__name__}: {exc}"}
        append_jsonl(checkpoint, result)
        count += 1
        print(f"[{count}/{len(tasks)}] {metric_type} {review_id}: {result.get('error', 'ok')}", flush=True)
    results = latest_records(read_jsonl(checkpoint))
    summarize(results, args.output)
    import torch
    package_versions = {}
    for package in ("faster-whisper", "ctranslate2", "facenet-pytorch", "torch", "av", "opencv-python"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "not-installed-as-package"
    try:
        commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                                check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {
        "experiment": "E2 automatic media quality proxies",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_commit": commit, "python": __import__("platform").python_version(),
        "packages": package_versions,
        "cuda_device": torch.cuda.get_device_name(0) if args.device == "cuda" and tasks else None,
        "asr_model": str(args.asr_model), "asr_model_directory_bytes": sum(
            p.stat().st_size for p in args.asr_model.rglob("*") if p.is_file()),
        "facenet_weights": str(args.facenet_weights),
        "facenet_weights_bytes": args.facenet_weights.stat().st_size,
        "compute_type": args.compute_type, "pack": str(args.pack),
        "device": args.device, "retry_failed": args.retry_failed,
        "script_sha256": file_sha256(Path(__file__)),
        "pack_csv_sha256": {p.name: file_sha256(p) for p in args.pack.glob("e2_*template.csv")},
    }
    (args.output / "e2_auto_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"new_records": count, "total_records": len(results),
                      "checkpoint": str(checkpoint), "summary": str(args.output / "e2_auto_summary.json"),
                      "csv": str(args.output / "e2_auto_metrics.csv"),
                      "manifest": str(args.output / "e2_auto_run_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
