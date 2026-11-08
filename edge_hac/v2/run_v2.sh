#!/bin/sh
# GW-HAC v2 driver.  Run from GW_HAC_Reproduction/ :   sh edge_hac/v2/run_v2.sh <stage> [<stage> ...]
#   test      unit tests of the v2 helpers (10 tests, < 1 min)
#   dev       development experiments 60-64 on the 25 study datasets (+ 60, 63 on the 11 old validation sets)
#   price     only the corrected price column (61b; needs the v2feat files of 61)
#   analyze   tables of the development runs (69) for dev and val11
#   power     delta, sigma_d, n for the pre-registration (70)
#   select    choose the confirmatory OpenML datasets from metadata, write PREREG_E7_FILLED.md (71)
#   fetch     download them -- only after OSF registration:  OSF_ID=10.17605/OSF.IO/XXXX sh ... fetch   (72)
#   confirm   confirmatory runs (73) and their pre-registered analysis (74)
#   timing    wall-clock timing of GW-HAC and comparators, ONE worker (75, 76); writes a small zip
#   pack      zip everything that has to be sent back (no data files)
# Environment: WORKERS (default 8), PYTHON (default .venv/bin/python, else python3).
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT/edge_hac"
WORKERS=${WORKERS:-8}
if [ -z "$PYTHON" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON=python3; fi
fi
OUT="$ROOT/results/GW-HAC_v2"
LOG="$OUT/logs"
mkdir -p "$LOG"
run() {  # run <log-name> <script> [args]
  name=$1; shift
  echo ">>> $name: $PYTHON $*"
  "$PYTHON" "$@" 2>&1 | tee -a "$LOG/$name.log"
}
for stage in "$@"; do
  case $stage in
    test)    run test -m pytest v2/tests -q ;;
    dev)     run 60_dev     v2/60_dev_factorial.py --workers "$WORKERS" --groups dev
             run 60_val11   v2/60_dev_factorial.py --workers "$WORKERS" --groups val11
             run 61_dev     v2/61_dev_theory.py    --workers "$WORKERS" --groups dev
             run 61b_price  v2/61b_price_fix.py    --workers "$WORKERS" --groups dev
             run 62_dev     v2/62_dev_interp.py    --workers "$WORKERS" --groups dev
             run 63_skew    v2/63_dev_skew.py      --workers "$WORKERS" --groups dev val11
             run 64_noise   v2/64_dev_noise.py     --workers "$WORKERS" --groups dev ;;
    price)   run 61b_price  v2/61b_price_fix.py    --workers "$WORKERS" --groups dev ;;
    analyze) run 69_dev     v2/69_dev_analyze.py --groups dev
             run 69_val11   v2/69_dev_analyze.py --groups val11
             run 69_all     v2/69_dev_analyze.py --groups dev val11 ;;
    power)   run 70_power   v2/70_power.py ;;
    select)  run 71_select  v2/71_select_validation.py ;;
    fetch)   [ -n "$OSF_ID" ] || { echo "set OSF_ID=<registration id> after uploading v2/docs/PREREG_E7_FILLED.md"; exit 1; }
             run 72_fetch   v2/72_fetch_validation.py --osf-id "$OSF_ID" ;;
    confirm) run 73_confirm v2/73_confirm_run.py --workers "$WORKERS"
             run 74_analyze v2/74_confirm_analyze.py ;;
    timing)  run 75_timing  v2/75_timing.py --workers 1 --groups dev val2
             run 76_timing  v2/76_timing_analyze.py
             STAMP=$(date +%Y%m%d_%H%M); cd "$ROOT"
             ZIP="$ROOT/GW-HAC_timing_$STAMP.zip"
             zip -qr "$ZIP" results/GW-HAC_v2/raw/v2time_*.csv results/GW-HAC_v2/tables/timing_* results/GW-HAC_v2/logs/75_*
             echo "send: $ZIP"; cd "$ROOT/edge_hac" ;;
    pack)    STAMP=$(date +%Y%m%d_%H%M)
             "$PYTHON" -c "import platform,sys,json,numpy,scipy,sklearn,pandas;print(json.dumps(dict(python=sys.version,platform=platform.platform(),numpy=numpy.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,pandas=pandas.__version__),indent=1))" > "$OUT/environment.json"
             ( cd "$OUT" && find raw -name '*.csv' | sort | xargs shasum -a 256 > SHA256SUMS.txt ) || true
             cd "$ROOT"
             ZIP="$ROOT/GW-HAC_v2_results_$STAMP.zip"
             zip -qr "$ZIP" results/GW-HAC_v2 \
                 $(ls edge_hac/data/validation2/*.json 2>/dev/null) \
                 $(ls edge_hac/v2/docs/PREREG_E7_FILLED.md 2>/dev/null)
             echo "send: $ZIP" ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
