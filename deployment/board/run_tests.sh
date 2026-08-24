#!/usr/bin/env bash
# Build and run every x86 golden test. Run from deployment/board:
#     wsl bash run_tests.sh
# Each test compares the C forward against vectors exported from PyTorch;
# a PASS means the board will compute the same function as the paper's model.
set -u
cd "$(dirname "$0")"

# Build every target explicitly. `make` alone only makes test_golden, so a
# bare `make` leaves the other binaries stale — which silently passed a
# FEATHer architecture change once, because test_batch was never relinked.
echo "=== build ==="
make -s build/test_golden build/test_stsf build/test_stsf_mlp \
        build/test_mixlinear build/test_dipe build/test_pers \
        build/test_fits build/test_batch "$@" 2>&1 | tail -20 || true

echo
echo "=== run ==="
fail=0
for t in build/test_*; do
    [ -x "$t" ] || continue
    printf '%-22s ' "$(basename "$t")"
    out=$("./$t" 2>&1)
    last=$(echo "$out" | grep -Ei 'PASS|FAIL|max' | tail -1)
    echo "${last:-(no verdict)}"
    echo "$out" | grep -qi fail && fail=1
done

echo
[ "$fail" = 0 ] && echo "ALL PASS" || echo "SOME FAILED"
exit "$fail"
