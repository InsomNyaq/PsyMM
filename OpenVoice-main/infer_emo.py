#!/usr/bin/env python3
"""Compatibility entry: use an immutable plan prepared by generate_multimodal.py."""
import sys
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'psymm_generation').is_dir())
sys.path.insert(0, str(ROOT))
from psymm_generation.cli import main

if __name__ == '__main__':
    sys.argv.insert(1, 'audio')
    raise SystemExit(main())
