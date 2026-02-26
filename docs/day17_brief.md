# Day 17 Brief — Repo Polish & CI

## Context
Days 1-16 complete. Block 4 finale. README done, TEA built, patent-code mapping complete. Now make the repo production-grade: proper metadata, CI pipeline, changelog, and final polish. After today, the repo is ready for external eyes.

**Hardware:** 2019 iMac i9-9900K, 16GB DDR4.
**M4 Pro Status:** Mac Mini M4 Pro 12/16 64GB 1TB ordered from Apple AU. Pickup Brisbane City store ~March 30, 2026. M2 Ultra Mac Studio (64GB, 800 GB/s) offer pending from Armidale NSW.
**Location:** Brisbane, Queensland, Australia.
**Doctrine:** No new computation. Configuration, metadata, and documentation only.
**Sprint goal:** All documentation prepared to Apple Core Quality engineering standards.

## The Standard

After today, `git clone` → `pip install -e ".[dev]"` → `pytest` → green. No warnings, no missing metadata, no ambiguity about what this repo is or who owns it. A GitHub Actions CI pipeline runs tests on every push. The changelog tells the story of the sprint.

## Today's Deliverables

### 1. `pyproject.toml` — Complete Project Metadata

Current pyproject.toml has basics. Upgrade to production-grade:

```toml
[project]
name = "terncore"
version = "0.4.0"
description = "Ternary execution engine for neural network inference"
readme = "README.md"
license = {text = "Proprietary"}
requires-python = ">=3.10"
authors = [
    {name = "Robert Lakelin"},
]
keywords = ["ternary", "quantization", "neural-network", "inference", "npu"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Private :: Do Not Upload",
]
```

**Critical:** Add `"Private :: Do Not Upload"` classifier to prevent accidental PyPI publication.

Also verify these sections are correct:
- `[project.scripts]` — tern-convert entry point (added Day 14)
- `[project.optional-dependencies]` — dev, transformers groups
- `[tool.pytest.ini_options]` — filterwarnings (added Day 14)
- `[tool.setuptools.packages.find]` — where = ["src"]

### 2. `.github/workflows/test.yml` — CI Pipeline

GitHub Actions workflow that runs on push to main and on PRs:

```yaml
name: Tests
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: macos-latest
    strategy:
      matrix:
        python-version: ["3.11"]
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev,transformers]"
      - name: Run tests
        run: pytest tests/ -v --tb=short
      - name: Build C kernels
        run: |
          cd src/terncore/csrc
          make clean && make
      - name: Run C tests
        run: |
          cd src/terncore/csrc
          make test
```

**Notes:**
- Use `macos-latest` not `ubuntu-latest` — the C kernels detect macOS for NEON/AVX2 and the resource.getrusage RSS measurement is platform-specific.
- Single Python version (3.11) for now. Add 3.10/3.12 matrix after M4 Pro validation.
- Do NOT include TinyLlama download tests — they'd timeout in CI and require HuggingFace credentials.

### 3. `CHANGELOG.md` — Sprint History

Every commit from Days 1-16 as a structured changelog:

```markdown
# Changelog

All notable changes to tern-core during the 20-day sprint.

## [0.4.0] - 2026-02-26 (Block 4: Documentation & Polish)

### Added
- Day 16: Patent-code mapping (14 patents → source + tests + evidence)
- Day 15: TEA template and TinyLlama-1.1B reference evaluation
- Day 14: README rewrite, Google-style docstrings, tern-convert CLI
- Day 13: Evidence consolidation (EVIDENCE_PACKAGE.md)

## [0.3.0] - 2026-02-25 (Block 3: Generalisation & Scaling)

### Added
- Day 12: Performance scaling curve (4 models × seq lengths × 3 modes)
- Day 11: Multi-model generalisation (5 architectures, Conv1D support)

## [0.2.0] - 2026-02-24 (Block 2: Format & Pipeline)

### Added
- Day 10: tern-convert CLI pipeline (end-to-end conversion)
- Day 9: Cached sparsity bitmap and zero-skip benchmark
- Day 8: PackedTernaryLinear with 2-bit storage
- Day 7: .tern-model v2 format (TernModelWriter/Reader)
- Day 6: .tern-model v1 format and memory profiling

## [0.1.0] - 2026-02-23 (Block 1: Core Engine)

### Added
- Day 5: Gradient sensitivity analysis and taxonomy
- Day 4: STE training proof-of-concept (45.8x improvement)
- Day 3: TernaryLinear drop-in replacement
- Day 2: Per-layer sensitivity analysis (155 layers)
- Day 1: TernaryQuantizer and initial test suite
```

Extract actual dates from git log if they differ. The version numbers align with the block structure.

### 4. `LICENSE` file

```
Copyright (c) 2026 Robert Lakelin. All rights reserved.

This software is proprietary and confidential. Unauthorised copying,
modification, distribution, or use of this software, via any medium,
is strictly prohibited.

Patent Notice: This software implements technology described in the
Synapticode patent portfolio. See docs/PATENT_CODE_MAP.md for details.
```

Short, clear, proprietary. No ambiguity. The open-source release (Move 7, Month 3-4) will switch this to MIT/Apache 2.0 for the Tier 1 toolkit, but for now the repo is private and proprietary.

### 5. `.gitignore` Audit

Verify .gitignore covers:
- `*.tern` / `*.tern-model` (model files shouldn't be in repo)
- `models/` directory
- `.venv/` / `venv/`
- `__pycache__/` / `*.pyc`
- `.pytest_cache/`
- `*.so` / `*.dylib` (compiled C extensions)
- `dist/` / `build/` / `*.egg-info`
- `.DS_Store`
- `wandb/` (experiment tracking)
- `*.csv` in root (but NOT in benchmarks/)

### 6. Version Tag

```bash
git tag v0.4.0-beta
git push --tags
```

## Implementation Order

1. **pyproject.toml upgrade** (10 min) — complete metadata, verify all sections
2. **CI workflow** (15 min) — write .github/workflows/test.yml
3. **CHANGELOG.md** (15 min) — extract from git log, structure by block
4. **LICENSE** (5 min) — proprietary notice
5. **.gitignore audit** (5 min) — verify coverage, add missing patterns
6. **Verify CI locally** (10 min) — run the exact commands from the workflow manually
7. **Final check** (10 min) — pip install -e ".[dev]" fresh, pytest, verify tern-convert works
8. **Commit + tag** (5 min)

## Exit Criteria
- [ ] pyproject.toml has complete metadata including Private classifier
- [ ] .github/workflows/test.yml exists and CI commands work locally
- [ ] CHANGELOG.md covers Days 1-16 structured by block
- [ ] LICENSE file exists with proprietary notice
- [ ] .gitignore covers all generated/compiled/model files
- [ ] `pip install -e ".[dev]"` works clean
- [ ] `pytest tests/ -v` shows 166+ passed with no warnings
- [ ] `tern-convert --help` works
- [ ] Version tagged v0.4.0-beta
- [ ] Commit pushed

## Time Budget
| Phase | Estimate |
|-------|----------|
| pyproject.toml | 10 min |
| CI workflow | 15 min |
| CHANGELOG.md | 15 min |
| LICENSE | 5 min |
| .gitignore audit | 5 min |
| Local CI verification | 10 min |
| Final check + commit | 15 min |
| **Total** | **~1.5 hours** |

## What NOT To Do
- Do NOT add type hints (mypy audit). Out of scope — future refinement.
- Do NOT add pre-commit hooks. Keep it simple for now.
- Do NOT add code coverage reporting. Tests pass = good enough today.
- Do NOT add multiple Python versions to CI matrix. 3.11 only for now.
- Do NOT add badge images to README. Clean text, no decoration.
- Do NOT modify any source code. Configuration and documentation only.
- Do NOT include model files or large test fixtures in the repo.

## What This Enables
- Day 18: KSGC draft references a CI-verified, versioned, professional repo
- Day 19: Rebellions outreach links to a repo that passes CI on every push
- Day 20: Sprint retrospective tags v0.4.0-beta as the sprint milestone
- Move 7 (Month 3-4): Open-source release starts from a clean, versioned base
- Apple evaluation: CI green badge, proper metadata, professional repo structure
