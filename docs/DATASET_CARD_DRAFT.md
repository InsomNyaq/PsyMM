# PsyMM dataset card (draft)

**Status:** preparation only. No Hugging Face dataset is uploaded or linked here. The paper's corpus totals are historical manuscript figures, not validated contents of a public download.

## Scope and sources

PsyMM is intended as a bilingual text, speech and talking-head video research resource for psychological support dialogue. Sources are ESConv, ExTES, OpenR1-Psy and Psych8k (English), plus PsyDial and PsyQA (Chinese). These collections differ in origin and rights. Generated audio/video is synthetic; AvaMERG supplies acted reference media for identity and emotion selection. Neither is a real person's measured psychological state.

## Proposed public record

The internal export is a list of turn records and embeds local absolute paths. Publication requires a transform to relative paths, privacy review, and permission for every included text/media item. Proposed fields:

| Field | Meaning |
| --- | --- |
| `sample_id`, `dialogue_id`, `turn_index` | Pseudonymous unit identifiers |
| `source`, `source_split`, `language`, `role` | Collection, audited split, language and role |
| `text` | Licensed and reviewed utterance text, when redistributable |
| `intended_emotion`, `emotion_origin` | Normalized label and its provenance |
| `persona_id` | Release-specific pseudonym with documented stability scope |
| `audio`, `video` | Relative paths to cleared media |
| `audio_sha256`, `video_sha256` | Byte-level integrity checks |
| `qc` | Technical checks and separately identified perceptual ratings |

This is a proposal, not the current export schema. Current records include `asset_id`, `dialogue_id`, `role`, `language`, `emotion`, `emotion_origin`, `reference`, `media`, and `perceptual_review_status`.

## Quality and limitations

Existence, hash and decode checks do not establish emotional correctness, natural speech, lip synchronization, consistent identity or counseling safety. Chinese audio quality is a known revision concern; Chinese generated media must not be described as validated until corrected generation and language-specific review are complete. Report failures and exclusions alongside successful examples by source, language, emotion and persona.

## Access and rights gates

Record permission for each source text, AvaMERG reference, generated portrait/audio/video and model output. Separate a public representative subset from reviewer-only material. The code's Apache license does not license data or model weights. Review text for personal information and faces/voices for consent and provenance; exclude items that cannot be redistributed. The final card should define intended research use and high-risk use restrictions.

## Evidence required before publication

The final card needs actual subset counts, split definitions, sampling seed, coverage and failures, technical and human quality results, hashes, version, layout, rights table, citation and an independently checked access link. None is claimed complete here.
