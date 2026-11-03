"""Resolve migrated encoder paths without editing pretrained checkpoint configs."""
import os
from pathlib import Path


def resolve_encoder_paths(config):
    root = Path(os.environ.get('PSYMM_WEIGHTS', Path(__file__).resolve().parents[4] / 'weights'))/'ola_weights'
    for field in ('speech_encoder', 'music_encoder'):
        original = getattr(config, field, None)
        override = os.environ.get('PSYMM_OLA_' + field.upper())
        value = override or original
        if not value:
            continue
        candidate = Path(value)
        if override:
            if not candidate.is_file():
                raise FileNotFoundError(f'{field}: {candidate}')
        elif (candidate.is_absolute() or str(value).startswith('/')) and not candidate.is_file():
            relocated = root/candidate.name
            if relocated.is_file():
                candidate = relocated
            if not candidate.is_file():
                raise FileNotFoundError(f'Missing {field}: {value}; set PSYMM_OLA_{field.upper()}')
        setattr(config,field,str(candidate) if candidate.is_absolute() or override else value)
    return config
