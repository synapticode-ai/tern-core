"""
Standalone CLI wrapper for tern-convert pipeline.

Converts a HuggingFace model to .tern-model format.

Usage:
    python tools/tern_convert.py TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output model.tern
    python tools/tern_convert.py MODEL_ID -o model.tern --threshold 0.7 --verify

See also: python -m terncore.convert (equivalent entry point)

Patents 10-12: Automated binary-to-ternary conversion pipeline.

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure tern-core is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from terncore.convert import main

if __name__ == "__main__":
    main()
