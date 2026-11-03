"""CPU-only preflight for frozen supplementary experiments.

The checker rejects legacy Ola exports and verifies the metadata contract for
new seeker-to-supporter inputs. It never loads model weights or media bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_rows(path: Path):
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("input must contain a JSON list or JSONL rows")
    return value


def basename(path):
    return Path(str(path)).stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--input", type=Path)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    deny = {str(Path(item["path"])): item for item in manifest["legacy_ola_denylist"]}
    result = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "denylisted_paths": len(deny),
        "input": None,
        "rows": 0,
        "checks": {},
        "status": "PASS",
    }
    if args.input is None:
        result["checks"]["rules_loaded"] = True
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    path = args.input.resolve()
    if str(path) in deny:
        raise SystemExit(f"REJECTED: legacy Ola export is denylisted: {path}")
    if not path.is_file():
        raise SystemExit(f"REJECTED: input does not exist: {path}")
    digest = sha256(path)
    if any(digest == item["sha256"] for item in deny.values()):
        raise SystemExit(f"REJECTED: input hash matches a legacy Ola export: {path}")
    rows = load_rows(path)
    checks = {"path_not_denylisted": True, "hash_not_denylisted": True,
              "input_is_seeker": 0, "target_is_supporter": 0,
              "input_precedes_target": 0, "target_media_not_bound": 0,
              "same_asset_rejected": 0}
    required = ("input_role", "target_role", "input_turn_index", "target_turn_index",
                "input_asset_id", "target_asset_id")
    for i, row in enumerate(rows):
        missing = [key for key in required if key not in row]
        if missing:
            raise SystemExit(f"REJECTED: row {i} missing binding fields: {', '.join(missing)}")
        if row.get("input_asset_id") == row.get("target_asset_id"):
            checks["same_asset_rejected"] += 1
            raise SystemExit(f"REJECTED: row {i} uses the same input and target asset")
        if "id" in row and any(basename(row.get(k, "")) == str(row["id"]) for k in ("audio", "video")):
            raise SystemExit(f"REJECTED: row {i} media basename equals target id")
        if row.get("input_role") == "seeker":
            checks["input_is_seeker"] += 1
        else:
            raise SystemExit(f"REJECTED: row {i} input_role is not seeker")
        if row.get("target_role") == "supporter":
            checks["target_is_supporter"] += 1
        else:
            raise SystemExit(f"REJECTED: row {i} target_role is not supporter")
        if not row["input_turn_index"] < row["target_turn_index"]:
            raise SystemExit(f"REJECTED: row {i} input is not before target")
        checks["input_precedes_target"] += 1
        if row.get("target_media") is not None or row.get("target_asset_id") in row.get("input_media_ids", []):
            raise SystemExit(f"REJECTED: row {i} target media is bound into the input")
        checks["target_media_not_bound"] += 1
    result.update({"input": str(path), "input_sha256": digest, "rows": len(rows), "checks": checks})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
