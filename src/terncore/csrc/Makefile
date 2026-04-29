# Makefile — libterncore shared library + test executables
#
# Detects platform (x86_64 vs arm64) and applies per-file SIMD flags.
# Detects OpenMP (libomp on macOS, system OpenMP on Linux).
#
# The shared library is built WITHOUT OpenMP to avoid libomp conflicts
# when loaded alongside PyTorch via ctypes.  OpenMP pragmas in the source
# are silently ignored when compiled without -fopenmp.  OpenMP threading
# is enabled via the PyTorch C++ extension (torch_bindings.cpp), which
# uses PyTorch's own libomp.
#
# Test executables ARE built with OpenMP since they don't load PyTorch.
#
# Usage:
#   make              Build shared library
#   make test         Build and run all C tests
#   make clean        Remove build artifacts
#
# Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.

CC       = cc
CFLAGS   = -std=c11 -O2 -Wall -Wextra -Wpedantic -fPIC
LDFLAGS  = -shared

UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)

# ── Platform-specific library name ──────────────────────────────────

ifeq ($(UNAME_S),Darwin)
    LIB = libterncore.dylib
else
    LIB = libterncore.so
endif

# ── OpenMP detection (used by test executables only) ────────────────

ifeq ($(UNAME_S),Darwin)
    OMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
    ifneq ($(OMP_PREFIX),)
        ifneq ($(wildcard $(OMP_PREFIX)/lib/libomp.dylib),)
            OMP_CFLAGS  = -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
            OMP_LDFLAGS = -L$(OMP_PREFIX)/lib -lomp
        endif
    endif
else
    OMP_CFLAGS  = -fopenmp
    OMP_LDFLAGS = -fopenmp
endif

# ── Source files ────────────────────────────────────────────────────

COMMON_SRCS = ternary_matmul.c ternary_packed.c sparse_skip.c bindings.c
SIMD_SRCS   =

# Architecture-specific SIMD sources and flags
ifeq ($(UNAME_M),x86_64)
    SIMD_SRCS  += ternary_avx2.c ternary_neon.c
    AVX2_FLAGS  = -mavx2
else ifneq (,$(filter $(UNAME_M),arm64 aarch64))
    SIMD_SRCS  += ternary_neon.c ternary_avx2.c
endif

ALL_SRCS = $(COMMON_SRCS) $(SIMD_SRCS)
OBJS     = $(ALL_SRCS:.c=.o)

# ── Version header generation from pyproject.toml ───────────────────
#
# Single source of truth: bumps in pyproject.toml flow to the C
# kernel automatically at build time. Uses tomllib (Python 3.11+) or
# falls back to tomli (Python 3.10) per pyproject.toml's
# requires-python = ">=3.10".

PYPROJECT      = ../../../pyproject.toml
VERSION_HEADER = terncore_version.h

.DEFAULT_GOAL := all

define VERSION_HEADER_SCRIPT
try:
    import tomllib
except ImportError:
    import tomli as tomllib
with open('$(PYPROJECT)', 'rb') as f:
    v = tomllib.load(f)['project']['version']
print('#ifndef TERNCORE_VERSION_H')
print('#define TERNCORE_VERSION_H')
print(f'#define TERNCORE_VERSION_STRING "{v}"')
print('#endif')
endef
export VERSION_HEADER_SCRIPT

$(VERSION_HEADER): $(PYPROJECT)
	@echo "Generating $@ from $<"
	@python3 -c "$$VERSION_HEADER_SCRIPT" > $@

# ── Targets ─────────────────────────────────────────────────────────

.PHONY: all clean test

all: $(VERSION_HEADER) $(LIB)

$(LIB): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

# ── Per-file compilation rules (NO OpenMP for shared library) ───────

ternary_avx2.o: ternary_avx2.c ternary_simd.h ternary_packed.h ternary_matmul.h
	$(CC) $(CFLAGS) $(AVX2_FLAGS) -c -o $@ $<

ternary_neon.o: ternary_neon.c ternary_simd.h ternary_packed.h ternary_matmul.h
	$(CC) $(CFLAGS) -c -o $@ $<

bindings.o: bindings.c sparse_skip.h ternary_simd.h ternary_packed.h ternary_matmul.h $(VERSION_HEADER)
	$(CC) $(CFLAGS) $(AVX2_FLAGS) -c -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# ── Test executables (WITH OpenMP — no PyTorch conflict) ────────────

ALL_C = $(COMMON_SRCS) $(SIMD_SRCS)

test_matmul: test_matmul.c ternary_matmul.c
	$(CC) $(CFLAGS) $(OMP_CFLAGS) $(OMP_LDFLAGS) -o $@ $^

test_packed: test_packed.c ternary_packed.c ternary_matmul.c
	$(CC) $(CFLAGS) $(OMP_CFLAGS) $(OMP_LDFLAGS) -o $@ $^

test_sparse_skip: test_sparse_skip.c $(ALL_C) $(VERSION_HEADER)
	$(CC) $(CFLAGS) $(OMP_CFLAGS) $(AVX2_FLAGS) $(OMP_LDFLAGS) -o $@ test_sparse_skip.c $(ALL_C)

test_simd: test_simd.c $(ALL_C)
	$(CC) $(CFLAGS) $(OMP_CFLAGS) $(AVX2_FLAGS) $(OMP_LDFLAGS) -o $@ $^

test: test_matmul test_packed test_sparse_skip test_simd
	@echo "=== test_matmul ===" && ./test_matmul
	@echo "=== test_packed ===" && ./test_packed
	@echo "=== test_sparse_skip ===" && ./test_sparse_skip
	@echo "=== test_simd ===" && ./test_simd
	@echo ""
	@echo "All C test suites passed."

# ── Cleanup ─────────────────────────────────────────────────────────

clean:
	rm -f $(OBJS) $(LIB) $(VERSION_HEADER) test_matmul test_packed test_sparse_skip test_simd
