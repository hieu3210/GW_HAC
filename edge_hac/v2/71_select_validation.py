"""Select the confirmatory OpenML datasets from METADATA ONLY (no data file is downloaded) and write the filled
pre-registration.

Uses the ``openml`` package if installed, otherwise the OpenML REST API (``requests``).  Reads tables/power.json
(script 70).  Writes
    data/validation2/SELECTION.json        all candidates, exclusions with reasons, the selected ids
    data/validation2/SELECTION_LOCK.json   SHA-256 of the selection and of the filled pre-registration
    v2/docs/PREREG_E7_FILLED.md            the document to upload to OSF (unchanged!)

    python v2/71_select_validation.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import v2lib as L

QUAL = ["NumberOfInstances", "NumberOfFeatures", "NumberOfNumericFeatures", "NumberOfClasses", "NumberOfMissingValues",
        "MajorityClassSize"]


def list_openml(snapshot: str | None = None) -> pd.DataFrame:
    if snapshot:                                        # offline: archived raw listing (exact reproduction)
        return pd.read_csv(snapshot, keep_default_na=False, dtype={"name": str, "status": str})
    try:
        import openml
        df = openml.datasets.list_datasets(output_format="dataframe", status="active")
        return df[["did", "name", "version", "status"] + [q for q in QUAL if q in df]].reset_index(drop=True)
    except ImportError:
        pass
    import requests
    rows, off, lim = [], 0, 5000
    while True:
        r = requests.get(f"https://www.openml.org/api/v1/json/data/list/status/active/limit/{lim}/offset/{off}", timeout=120)
        if r.status_code != 200:
            break
        ds = r.json().get("data", {}).get("dataset", [])
        if not ds:
            break
        for d in ds:
            q = {x["name"]: x["value"] for x in d.get("quality", [])}
            rows.append(dict(did=int(d["did"]), name=d["name"], version=int(d["version"]), status=d["status"],
                             **{k: q.get(k) for k in QUAL}))
        off += lim
    if not rows:
        raise RuntimeError("OpenML listing failed (no network?)")
    return pd.DataFrame(rows)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def family(name: str) -> str:
    """Name family: the first token of the lower-cased name (split at _ - . ( ) or space) if it has >= 3 characters."""
    t = re.split(r"[_\-\s.()]+", str(name).lower())
    return t[0] if t and len(t[0]) >= 3 else str(name).lower()


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(L.HERE.glob("*.py")) + [L.HERE / "protocol_v2.json"]:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(L.RESULTS))
    ap.add_argument("--listing", default=None, help="archived raw OpenML listing CSV (offline, reproducible)")
    a = ap.parse_args()
    S = L.CFG["selection"]
    power = json.loads((Path(a.out) / "tables" / "power.json").read_text())
    df = list_openml(a.listing)
    for q in QUAL:
        df[q] = pd.to_numeric(df[q], errors="coerce")
    df = df[df.status == "active"].copy()
    df["nname"] = df.name.map(norm)
    df = df.sort_values(["nname", "version"]).groupby("nname", as_index=False).tail(1)   # latest active version
    excl_names = [norm(n) for n in S["exclude_names"]]
    reasons = {}

    def reason(r):
        nn = r.nname
        if int(r.did) in S["exclude_ids"]:
            return "used id"
        if nn in excl_names or any(len(e) >= 5 and nn.startswith(e) for e in excl_names):
            return "used name"
        if any(p in str(r["name"]).lower() for p in S["exclude_name_patterns"]):
            return "pattern"
        N, F, Fn, K = r.NumberOfInstances, r.NumberOfFeatures, r.NumberOfNumericFeatures, r.NumberOfClasses
        if not all(np.isfinite([N, F, Fn, K])):
            return "metadata missing"
        if not (S["K_min"] <= K <= S["K_max"]):
            return "classes"
        if not (S["N_min"] <= N <= S["N_max"]):
            return "N"
        if not (S["F_min"] <= Fn <= S["F_max"]):
            return "numeric features"
        if not np.isfinite(r.MajorityClassSize) or r.MajorityClassSize / N > S["majority_max"]:
            return "majority"
        if np.isfinite(r.NumberOfMissingValues) and r.NumberOfMissingValues / (N * F) > S["missing_max"]:
            return "missing"
        return ""

    df["reason"] = df.apply(reason, axis=1)
    # --- v2.1 rules (protocol_v2.json -> selection; see GW_HAC_v2.1_Results/README.md) -------------------------
    def shape(r):
        return (int(r.NumberOfInstances), int(r.NumberOfFeatures), int(r.NumberOfClasses))

    used = df[df.reason.isin(["used id", "used name"])].dropna(
        subset=["NumberOfInstances", "NumberOfFeatures", "NumberOfClasses"])
    used_shapes = {shape(r) for _, r in used.iterrows()}
    used_fams = {family(n) for n in S["exclude_names"]} | set(used.name.map(family))
    cand = df.reason == ""
    if S.get("numeric_share_min"):
        low = cand & (df.NumberOfNumericFeatures / (df.NumberOfFeatures - 1).clip(lower=1) < S["numeric_share_min"])
        df.loc[low, "reason"] = "numeric share"
    if S.get("exclude_used_shapes"):
        cand = df.reason == ""
        m = cand & df.apply(lambda r: cand[r.name] and shape(r) in used_shapes, axis=1)
        df.loc[m, "reason"] = "same (N,F,K) as a used dataset"
    if S.get("exclude_used_families"):
        m = (df.reason == "") & df.name.map(family).isin(used_fams)
        df.loc[m, "reason"] = "family of a used dataset"
    if S.get("dedup_exact"):
        c = df[df.reason == ""].sort_values("did")
        cols = S.get("dedup_key", ["NumberOfInstances", "NumberOfFeatures", "NumberOfClasses", "MajorityClassSize"])
        key = c.apply(lambda r: tuple(int(r[k]) for k in cols), axis=1)
        df.loc[c.index[key.duplicated(keep="first").to_numpy()], "reason"] = "duplicate of a lower id"
    manual = {int(k): v for k, v in S.get("exclude_ids_v2_1", {}).items()}
    m = (df.reason == "") & df.did.astype(int).isin(manual)
    df.loc[m, "reason"] = "manual: copy of another candidate (protocol exclude_ids_v2_1)"
    if S.get("one_per_family"):
        c = df[df.reason == ""].sort_values("did")
        rng_f = np.random.default_rng(S["seed"])
        keep = {int(rng_f.choice(g.did.to_numpy())) for _, g in c.groupby(c.name.map(family), sort=True)}
        df.loc[c.index[~c.did.isin(keep).to_numpy()], "reason"] = "another member of its name family"
    folder = L.VAL2_DIR
    folder.mkdir(parents=True, exist_ok=True)
    listing = folder / "OPENML_LISTING.csv"                       # snapshot of the metadata the selection used
    df.drop(columns=["nname"]).sort_values("did").to_csv(listing, index=False)
    # ------------------------------------------------------------------------------------------------------------
    q = df[df.reason == ""].sort_values("did")
    n_max = int(power["n_max"])
    if len(q) > n_max:
        pick = np.sort(np.random.default_rng(S["seed"]).choice(len(q), n_max, replace=False))
        sel = q.iloc[pick]
    else:
        sel = q
    mode = "test" if len(sel) >= power["n_required"] else "estimation"
    ids = [int(x) for x in sel.did]
    sel_sha = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
    used = df[df.reason.isin(["used id", "used name"])][["did", "name", "version", "reason"]]
    folder = L.VAL2_DIR
    folder.mkdir(parents=True, exist_ok=True)
    selection = dict(created=dt.datetime.now().isoformat(timespec="seconds"), n_listed=int(len(df)),
                     n_qualify=int(len(q)), n_selected=len(ids), mode=mode, seed=S["seed"], ids=ids, sha256=sel_sha,
                     selected=sel[["did", "name", "version"] + QUAL].to_dict("records"),
                     excluded_used=used.to_dict("records"),
                     reason_counts=df.reason.replace("", "qualifies").value_counts().to_dict(), power=power)
    (folder / "SELECTION.json").write_text(json.dumps(selection, indent=1, default=float))
    tpl = (L.HERE / "docs" / "PREREG_E7_TEMPLATE.md").read_text()
    lst = "\n".join(f"- {int(r.did)} {r['name']} v{int(r.version)} (N={int(r.NumberOfInstances)}, "
                    f"F_num={int(r.NumberOfNumericFeatures)}, K={int(r.NumberOfClasses)})" for _, r in sel.iterrows())
    exm = "\n".join(f"- {int(r.did)} {r['name']} ({r.reason})" for _, r in used.iterrows()) or "- none"
    fill = dict(GENERATED=selection["created"], SELF_SHA_NOTE="see SELECTION_LOCK.json", CODE_HASH=code_hash()[:16],
                N_PATTERNS=len(S["exclude_name_patterns"]), N_QUALIFY=len(q), N_MAX=n_max, SEL_SEED=S["seed"],
                N_SELECTED=len(ids), DELTA=f"{power['delta']:.3f}", DELTA_RAW=f"{power['delta_raw']:.4f}",
                N_DEV=power["n_dev"], SIGMA_D=f"{power['sigma_d']:.4f}", N_REQ=power["n_required"], MODE=mode, N_FORMULA=power.get("n_formula_not_used", "--"), SIM_SEED=power.get("sim_seed", "--"),
                BOOT_SEED=L.CFG["bootstrap"]["seed"], EXCLUDED_MATCHES=exm, SELECTED_LIST=lst, SEL_SHA=sel_sha)
    for k, v in fill.items():
        tpl = tpl.replace("{{" + k + "}}", str(v))
    out_md = L.HERE / "docs" / "PREREG_E7_FILLED.md"
    out_md.write_text(tpl)
    lock = dict(prereg_file=str(out_md.relative_to(L.REPO)), prereg_sha256=hashlib.sha256(out_md.read_bytes()).hexdigest(),
                selection_sha256=sel_sha, code_sha256=code_hash(), created=selection["created"], osf_id=None,
                listing_file=(str(listing.relative_to(L.REPO)) if listing.is_relative_to(L.REPO) else str(listing)), listing_sha256=hashlib.sha256(listing.read_bytes()).hexdigest())
    (folder / "SELECTION_LOCK.json").write_text(json.dumps(lock, indent=1))
    print(f"listed {len(df)}, qualify {len(q)}, selected {len(ids)} (n_required {power['n_required']}, mode {mode})")
    print(f"pre-registration: {out_md}\nSHA-256: {lock['prereg_sha256']}")
    print("Review 'excluded_used' in SELECTION.json, then upload PREREG_E7_FILLED.md to OSF BEFORE script 72.")


if __name__ == "__main__":
    main()
