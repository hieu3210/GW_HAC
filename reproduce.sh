#!/bin/sh
# Reproduction driver for "Distortion Analysis of Hedge-Algebra Quantization for Multigranular Clustering".
# Run from this folder:   sh reproduce.sh [stage ...]
#
#   setup          create .venv with the pinned environment and the ExKMC baseline (needs internet access)
#   verify         SHA-256 of the data, of the archived results and of the code frozen before the confirmatory run
#   test           unit tests of the experiment library (10 tests, a few seconds)
#   numerics       Section 6: Table 1, Fig. 1 and the in-text numbers (no data needed, about 1 minute)
#   tables         Section 7: regenerate Tables 2, 3, 5, Fig. 2, the Iris example (Table 4), the timing numbers
#                  and the supplementary tables from the archived per-split results, into out/ (about 1 minute)
#   check          compare out/ with the archive (byte for byte) and with the numbers printed in the paper
#   spot           recompute the per-split results of one development dataset (DS, default iris) and one
#                  confirmatory dataset (CONF, default 40705_tokyo1) and compare them with the archive (about 30 s)
#   selection      redo the confirmatory dataset selection offline from the archived OpenML metadata snapshot,
#                  in a sandbox, and compare the selected ids with the registered selection (about 30 s)
#   rerun-dev      recompute all development results into results/GW-HAC_v2 (hours; WORKERS processes)
#   rerun-confirm  recompute all confirmatory results into results/GW-HAC_v2 (hours)
#   rerun-timing   single-core timing of all methods into results/GW-HAC_v2 (15-40 min; machine dependent)
#
# Without arguments: verify test numerics tables check.
# Environment: PYTHON (interpreter; default .venv/bin/python, else python3), WORKERS (default 4), DS, CONF.
set -e
cd "$(dirname "$0")"
ROOT=$(pwd)
PY=${PYTHON:-$ROOT/.venv/bin/python}
[ -x "$PY" ] || PY=$(command -v python3)
ARCH=$ROOT/results/archive
OUT=$ROOT/out
W=${WORKERS:-4}
STAGES=${*:-verify test numerics tables check}

for s in $STAGES; do
  echo "== $s"
  case $s in
  setup)
    python3 -m venv .venv
    .venv/bin/python -m pip install -q --upgrade pip
    .venv/bin/python -m pip install -q -r requirements.txt
    .venv/bin/python tools/install_exkmc.py            # IMM / ExKMC threshold-tree baselines
    PY=$ROOT/.venv/bin/python ;;
  verify)
    shasum -a 256 -c --quiet edge_hac/data/DATA_SHA256SUMS.txt && echo "data: all checksums OK"
    (cd "$ARCH" && shasum -a 256 -c --quiet SHA256SUMS.txt) && echo "archived results: all checksums OK"
    echo "code frozen before the confirmatory run (expected differences: SELECTION_LOCK.json, which records the OSF id"
    echo "and the download times, and run_v2.sh, which gained the 'timing' stage afterwards; the frozen copy is"
    echo "edge_hac/v2/docs/run_v2.sh.v2.2_lock):"
    shasum -a 256 -c edge_hac/v2/docs/LOCK_MANIFEST_v2.2.sha256 2>/dev/null | grep -v ": OK$" || true
    "$PY" - <<'EOF'
import hashlib, json
lock = json.load(open("edge_hac/data/validation2/SELECTION_LOCK.json"))
sel = json.load(open("edge_hac/data/validation2/SELECTION.json"))
pre = hashlib.sha256(open("edge_hac/v2/docs/PREREG_E7_FILLED.md", "rb").read()).hexdigest()
ids = hashlib.sha256(",".join(map(str, sel["ids"])).encode()).hexdigest()
print("registered document unchanged:", pre == lock["prereg_sha256"], "| selection unchanged:", ids == lock["selection_sha256"],
      "| OSF:", lock["osf_id"])
EOF
    ;;
  test)
    (cd edge_hac && "$PY" -m pytest -q -p no:cacheprovider v2/tests) ;;
  numerics)
    (cd edge_hac/v2 && "$PY" paper_tools/numerics_sec6.py --out "$OUT") ;;
  tables)
    R=$OUT/results; rm -rf "$R"; mkdir -p "$R/tables" "$OUT/tables" "$OUT/figures"
    cp -R "$ARCH/raw" "$ARCH/logs" "$R/"
    cd edge_hac
    run() { echo "-- $*"; "$PY" "$@" > "$R/logs/regen_$(basename "$1" .py).log" 2>&1 || { tail -20 "$R/logs/regen_$(basename "$1" .py).log"; exit 1; }; }
    run v2/69_dev_analyze.py --groups dev --out "$R"
    run v2/69_dev_analyze.py --groups val11 --out "$R"
    run v2/69_dev_analyze.py --groups dev val11 --out "$R"
    run v2/70_power.py --out "$R"
    run v2/74_confirm_analyze.py --out "$R"
    run v2/76_timing_analyze.py --out "$R"
    cd v2
    "$PY" paper_tools/timing_numbers.py "$R" > "$OUT/timing_numbers.txt"
    "$PY" paper_tools/fig_confirm.py "$R" "$OUT/figures/confirm_scatter.pdf"
    "$PY" paper_tools/make_supp.py "$R" "$OUT/supplement.tex"
    "$PY" paper_tools/worked_example.py iris > "$OUT/worked_example_iris.txt"
    cd "$ROOT"
    cp "$R"/tables/* "$OUT/tables/"
    echo "tables in out/tables, figures in out/figures, out/supplement.tex, out/timing_numbers.txt, out/worked_example_iris.txt" ;;
  check)
    "$PY" tools/check_outputs.py "$OUT" ;;
  spot)
    D=${DS:-iris}; C=${CONF:-40705_tokyo1}; SP=$OUT/spot; rm -rf "$SP"; mkdir -p "$SP"
    (cd edge_hac
     for x in 60_dev_factorial 61_dev_theory 61b_price_fix 62_dev_interp 63_dev_skew 64_dev_noise; do
       echo "-- $x $D"; "$PY" v2/$x.py --workers "$W" --groups dev --only "$D" --out "$SP" > "$SP/$x.log" 2>&1
     done
     echo "-- 73_confirm_run $C"; "$PY" v2/73_confirm_run.py --workers "$W" --groups val2 --only "$C" --out "$SP" > "$SP/73.log" 2>&1)
    "$PY" tools/compare_raw.py "$SP/raw" "$ARCH/raw" ;;
  selection)
    SB=$OUT/selection; rm -rf "$SB"; mkdir -p "$SB/edge_hac/data" "$SB/res/tables" "$SB/val2"
    cp -R edge_hac/v2 "$SB/edge_hac/"
    for x in edgehac experiments configs; do ln -s "$ROOT/edge_hac/$x" "$SB/edge_hac/$x"; done
    ln -s "$ROOT/edge_hac/data/edge" "$SB/edge_hac/data/edge"; ln -s "$ROOT/data" "$SB/data"; ln -s "$ROOT/haclustering" "$SB/haclustering"
    cp "$ARCH/tables/power.json" "$SB/res/tables/"
    (cd "$SB/edge_hac" && GWHAC_VAL2_DIR="$SB/val2" "$PY" v2/71_select_validation.py \
        --listing "$ROOT/edge_hac/data/validation2/OPENML_RAW_LISTING_2026-11-06.csv" --out "$SB/res" | head -1)
    "$PY" - "$SB/val2/SELECTION.json" <<'EOF'
import json, sys
new, old = json.load(open(sys.argv[1])), json.load(open("edge_hac/data/validation2/SELECTION.json"))
lock = json.load(open("edge_hac/data/validation2/SELECTION_LOCK.json"))
ok = new["ids"] == old["ids"] and new["sha256"] == lock["selection_sha256"]
print(f"qualifying {new['n_qualify']} (registered {old['n_qualify']}), selected {len(new['ids'])}; "
      f"ids identical to the registered selection: {ok}")
sys.exit(0 if ok else 1)
EOF
    ;;
  rerun-dev)
    WORKERS=$W PYTHON=$PY sh edge_hac/v2/run_v2.sh test dev analyze power ;;
  rerun-confirm)
    WORKERS=$W PYTHON=$PY sh edge_hac/v2/run_v2.sh confirm ;;
  rerun-timing)
    PYTHON=$PY sh edge_hac/v2/run_v2.sh timing
    (cd edge_hac/v2 && "$PY" paper_tools/timing_numbers.py "$ROOT/results/GW-HAC_v2") ;;
  *) echo "unknown stage $s"; exit 1 ;;
  esac
done
