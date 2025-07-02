#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------
# 0) Build directory
mkdir -p build

# -------------------------------------------------------------
# 1) Compile AXPY benchmark
gcc -O3 -march=rv64gc_xtheadvector -mabi=lp64d \
    -std=c99 -fgnu89-inline -Wall -pedantic \
    -I../../include \
    -o build/axpy_perf \
    ../../src/vectorized.c \
    ../../src/common.c \
    axpy_perf.c

# -------------------------------------------------------------
# 2) Vector sizes to test
sizes=(1024 2048 4096 8192 16384 32768 65536 \
       131072 262144 524288 1048576)

# -------------------------------------------------------------
# 3) Final unified CSV
out="axpy_perf_full.csv"
echo "n,time_serial,time_vectorized,speedup_time,cycles_serial,cycles_vector,"\
"speedup_cycles,pass,L1-loads,L1-misses,LLC-loads,LLC-misses" > "$out"

# -------------------------------------------------------------
# 4) Loop over sizes
for n in \"${sizes[@]}\"; do
    echo \"Profiling n=$n ...\"

    # (a) — run program, capture its CSV line in a temp file
    tmp_prog_csv=$(mktemp)
    AXPY_CSV=\"$tmp_prog_csv\" ./build/axpy_perf \"$n\"
    prog_line=$(tail -n 1 \"$tmp_prog_csv\")
    rm \"$tmp_prog_csv\"

    # (b) — run perf and extract cache counters
    read l1_loads l1_miss llc_loads llc_miss <<< \"$(
        perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \\
                 -x, -- ./build/axpy_perf \"$n\" 2>&1 |
        awk -F',' '
            $3==\"L1-dcache-loads\"       { gsub(/[^0-9]/,\"\",$1); l1=$1 }
            $3==\"L1-dcache-load-misses\" { gsub(/[^0-9]/,\"\",$1); l1m=$1 }
            $3==\"LLC-loads\"             { gsub(/[^0-9]/,\"\",$1); llc=$1 }
            $3==\"LLC-load-misses\"       { gsub(/[^0-9]/,\"\",$1); llcm=$1 }
            END { print l1, l1m, llc, llcm }
        ')\"

    # (c) — append unified line
    echo \"${prog_line},${l1_loads},${l1_miss},${llc_loads},${llc_miss}\" >> \"$out\"
done

echo \"Profiling completed — results written to $out\"
