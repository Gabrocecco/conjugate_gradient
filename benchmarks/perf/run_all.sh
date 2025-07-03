#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Run all performance benchmarks in sequence:
# 1. AXPY benchmark
# 2. FOAM-based matrix-vector benchmark
# 3. Random matrix-vector benchmark
# ============================================================================

echo "============================================================"
echo "       Starting full benchmark suite: AXPY + FOAM + RANDOM"
echo "============================================================"
echo

# --- AXPY benchmark ---
echo "Running: run_perf_axpy.sh ..."
./run_perf_axpy.sh
echo "Finished: run_perf_axpy.sh"
echo

# --- FOAM benchmark ---
echo "Running: run_perf_foam.sh ..."
./run_perf_foam.sh
echo "Finished: run_perf_foam.sh"
echo

# --- RANDOM benchmark ---
echo "Running: run_perf_random.sh ..."
./run_perf_random.sh
echo "Finished: run_perf_random.sh"
echo

echo "All benchmarks completed successfully."
