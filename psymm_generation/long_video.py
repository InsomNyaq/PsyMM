"""Bounded DICE calls with audio context, exact frame trims and resumable parts.

The full original voice track is used for the final mux. No text is dropped,
and portrait/identity/emotion stay fixed. Boundary motion still needs review.
"""
import json
import math
import os
import subprocess
from pathlib import Path

from .core import atomic_json, checked, digest, file_hash, probe_video


def windows(samples, sample_rate, seconds=20, context_seconds=1):
    if samples <= 0 or sample_rate <= 0 or seconds < 2 or context_seconds < 0:
        raise ValueError('Invalid segment bounds')

    if int(seconds) != seconds or int(context_seconds) != context_seconds:
        raise ValueError('Segment and context seconds must be integers')
    step, context = int(seconds * sample_rate), int(context_seconds * sample_rate)
    return [dict(index=i, start=start, end=min(start + step, samples),
                 context_start=max(0, start-context),
                 context_end=min(samples, start+step+context),



                 right_padding_samples=sample_rate if start + step >= samples else 0)
            for i, start in enumerate(range(0, samples, step))]


def video_info(path):
    return json.loads(checked(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                              '-show_streams', '-of', 'json', path]).stdout)['streams'][0]


def check_part(path, frames):
    v = video_info(path)
    if v['avg_frame_rate'] != '25/1' or int(v['nb_frames']) != frames:
        raise ValueError(f'Wrong segment fps/frame count: {path}')
    checked(['ffmpeg', '-v', 'error', '-xerror', '-i', path, '-f', 'null', '-'])
    return (v['width'], v['height'], v['codec_name'], v.get('pix_fmt'))


def segmented_video(model, audio, portrait, prior, output, cache, options, seed):
    import soundfile as sf
    audio, output, cache = Path(audio), Path(output), Path(cache)
    info = sf.info(str(audio))
    parts = windows(info.frames, info.samplerate, options['video_segment_seconds'])
    identity = dict(audio_sha256=file_hash(audio), portrait_sha256=file_hash(portrait),
                    prior_sha256=file_hash(prior), configuration_hash=options['_configuration_hash'],
                    seed=seed, seconds=options['video_segment_seconds'], context_seconds=1,
                    sample_rate=info.samplerate, samples=info.frames, fps=25)
    cache = cache / digest(identity)
    cache.mkdir(parents=True, exist_ok=True)
    atomic_json(cache / 'plan.json', dict(identity=identity, parts=parts))
    completed, geometry = [], None
    for part in parts:
        index = part['index']
        path = cache / f'part_{index:05d}.mp4'
        receipt = path.with_suffix('.json')
        frames = math.ceil((part['end']-part['start'])*25/info.samplerate)
        reusable = False
        if path.exists() and receipt.exists():
            try:
                saved = json.loads(receipt.read_text())
                reusable = saved['sha256'] == file_hash(path) and saved['part'] == part
                if reusable:
                    shape = check_part(path, frames)
            except (ValueError, KeyError, OSError, RuntimeError, subprocess.CalledProcessError):
                reusable = False
        if not reusable:
            wav = cache / f'context_{index:05d}.wav'
            raw = cache / f'raw_{index:05d}.mp4'
            tmp = path.with_name(path.stem + '.partial.mp4')
            try:
                with sf.SoundFile(str(audio)) as src:
                    src.seek(part['context_start'])
                    data = src.read(part['context_end']-part['context_start'], dtype='float32', always_2d=True)
                if part['right_padding_samples']:
                    import numpy as np
                    data = np.pad(data, ((0, part['right_padding_samples']), (0, 0)))
                sf.write(str(wav), data, info.samplerate, subtype='PCM_16')
                chunk_seed = int(digest([seed, index])[:8], 16) % (2**31)
                result = model.process(str(portrait), str(wav), str(prior), str(raw),
                                       min_resolution=512, inference_steps=options['steps'],
                                       ref_scale=options['ref_scale'], emo_scale=options['emo_scale'],
                                       seed=chunk_seed)
                if result != 0:
                    raise RuntimeError(f'DICE segment {index} failed: {result}')
                probe_video(raw, len(data)/info.samplerate, decode=True)
                if video_info(raw)['avg_frame_rate'] != '25/1':
                    raise ValueError('Expected 25 fps DICE output')
                start_frame = (part['start']-part['context_start'])*25//info.samplerate
                if int(video_info(raw)['nb_frames']) < start_frame + frames:
                    raise ValueError('DICE segment is shorter than the retained frame range')
                checked(['ffmpeg', '-v', 'error', '-y', '-i', raw, '-an', '-vf',
                         f'trim=start_frame={start_frame}:end_frame={start_frame+frames},setpts=PTS-STARTPTS',
                         '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '2', tmp])
                shape = check_part(tmp, frames)
                os.replace(tmp, path)
                atomic_json(receipt, dict(part=part, sha256=file_hash(path), frames=frames))
            finally:
                for temporary in (wav, raw, tmp, raw.with_name(raw.stem+'_noaudio.mp4')):
                    if temporary.exists():
                        temporary.unlink()
        if geometry is not None and geometry != shape:
            raise ValueError('Inconsistent segment geometry/codec')
        geometry = shape
        completed.append(dict(part=part, path=str(path), sha256=file_hash(path), frames=frames))
        print(f'SEGMENT {index+1}/{len(parts)} {"RESUME" if reusable else "COMPLETE"} {output.stem}', flush=True)
    listing = cache / 'concat.txt'

    listing.write_text(''.join(f"file 'part_{p['index']:05d}.mp4'\n" for p in parts))
    tmp = output.with_name(output.stem+'.partial.mp4')
    try:




        checked(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '1', '-i', listing,
                 '-i', audio, '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'libx264',
                 '-crf', '18', '-pix_fmt', 'yuv420p', '-r', '25', '-threads', '2',
                 '-c:a', 'aac', '-movflags', '+faststart', tmp])
        check_part(tmp, math.ceil(info.frames*25/info.samplerate))
        qc = probe_video(tmp, info.frames/info.samplerate, decode=True)
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink()
    return dict(qc=qc, segmented=True, segment_manifest=str(cache/'plan.json'),
                segments=completed, original_audio_sha256=identity['audio_sha256'],
                boundary_policy='1s audio context trimmed; no crossfade or speech truncation; visual review pending')
