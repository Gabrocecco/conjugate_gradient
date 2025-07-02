#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# Paths
proj_root="$(cd "$(dirname "$0")/../.." && pwd)"
build_dir="$proj_root/build"
src_dir="$proj_root/src"
inc_dir="$proj_root/include"
data_dir="$proj_root/benchmarks/perf/data"

mkdir -p "$build_dir" "$data_dir"

# ------------------------------------------------------------------
# Compile axpy_perf
gcc -O3 -std=c11 -march=rv64gc_xtheadvector -mabi=lp64d \
    -Wall -pedantic -I"$inc_dir" \
    -o "$build_dir/axpy_perf" \
    "$src_dir/vectorized.c" "$src_dir/common.c" axpy_perf.c

# ------------------------------------------------------------------
# Sizes to test
sizes=(1024 2048 4096 8192 16384 32768 65536 \
       131072 262144 524288 1048576)

# ------------------------------------------------------------------
# CSV finale (sovrascrive sempre)
out="$data_dir/axpy_perf_full.csv"
echo "n,time_serial,time_vectorized,speedup_time,cycles_serial,cycles_vector,"\
"speedup_cycles,pass,L1-loads,L1-misses,LLC-loads,LLC-misses" > "$out"

# ------------------------------------------------------------------
# Loop
for n in "${sizes[@]}"; do
    echo "Profiling n=$n ..."

    # (a) Prima esecuzione: otteniamo la riga tempi/cicli
    tmp_csv=$(mktemp)
    AXPY_CSV="$tmp_csv" "$build_dir/axpy_perf" "$n"
    prog_line=$(tail -n 1 "$tmp_csv")
    rm "$tmp_csv"

    # (b) Seconda esecuzione sotto perf (nessun CSV creato)
    read l1_loads l1_miss llc_loads llc_miss <<< "$(
        AXPY_CSV=/dev/null \
        perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
                 -x, -- "$build_dir/axpy_perf" "$n" 2>&1 |
        awk -F',' '
            $3=="L1-dcache-loads"       { gsub(/[^0-9]/,"",$1); l1=$1 }
            $3=="L1-dcache-load-misses" { gsub(/[^0-9]/,"",$1); l1m=$1 }
            $3=="LLC-loads"             { gsub(/[^0-9]/,"",$1); llc=$1 }
            $3=="LLC-load-misses"       { gsub(/[^0-9]/,"",$1); llcm=$1 }
            END { print l1, l1m, llc, llcm }
        ')"

    # (c) Scriviamo la riga unificata
    echo "${prog_line},${l1_loads},${l1_miss},${llc_loads},${llc_miss}" >> "$out"
done

echo "Profilazione completata — risultati in $out"
