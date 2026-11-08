"""Download the selected confirmatory datasets -- ONLY after the pre-registration is time-stamped on OSF.

Refuses to run unless
  * --osf-id is given (the OSF registration identifier / DOI), and
  * v2/docs/PREREG_E7_FILLED.md still has the SHA-256 recorded in data/validation2/SELECTION_LOCK.json.
Non-numeric columns are dropped; missing values are kept (imputed per split in 73).  A dataset that fails to download,
has no numeric column or fewer than 2 classes is excluded and NOT replaced.

Output: data/validation2/<name>.npz (X, y) and data/validation2/MANIFEST.json

    python v2/72_fetch_validation.py --osf-id 10.17605/OSF.IO/XXXXX
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import time

import numpy as np

import v2lib as L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--osf-id", required=True)
    a = ap.parse_args()
    folder = L.VAL2_DIR
    lock = json.loads((folder / "SELECTION_LOCK.json").read_text())
    pre = L.REPO / lock["prereg_file"]
    sha = hashlib.sha256(pre.read_bytes()).hexdigest()
    if sha != lock["prereg_sha256"]:
        raise SystemExit(f"STOP: {pre} changed after selection (sha {sha[:12]} != {lock['prereg_sha256'][:12]}).")
    sel = json.loads((folder / "SELECTION.json").read_text())
    if hashlib.sha256(",".join(map(str, sel["ids"])).encode()).hexdigest() != lock["selection_sha256"]:
        raise SystemExit("STOP: SELECTION.json changed after the lock.")
    if lock.get("osf_id") and lock["osf_id"] != a.osf_id:
        raise SystemExit(f"STOP: lock already records OSF id {lock['osf_id']}.")
    lock["osf_id"], lock["fetch_started"] = a.osf_id, dt.datetime.now().isoformat(timespec="seconds")
    (folder / "SELECTION_LOCK.json").write_text(json.dumps(lock, indent=1))
    from sklearn.datasets import fetch_openml
    mpath = folder / "MANIFEST.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {"osf_id": a.osf_id, "datasets": {}}
    for rec in sel["selected"]:
        did = int(rec["did"])
        name = f"{did}_" + re.sub(r"[^A-Za-z0-9]+", "_", str(rec["name"])).strip("_")[:40]
        if name in man["datasets"]:
            continue
        d = None
        for attempt in range(5):
            try:
                d = fetch_openml(data_id=did, as_frame=True, parser="auto")
                break
            except Exception as e:
                print(f"{name}: retry {attempt} ({type(e).__name__}: {e})", flush=True)
                time.sleep(5)
        if d is None:
            man["datasets"][name] = dict(id=did, dropped="download failed")
        else:
            num = d.data.select_dtypes(include=[np.number])
            X = num.to_numpy(dtype=float)
            yv = np.asarray(d.target).astype(str) if d.target is not None else None
            if yv is None or X.shape[1] == 0:
                man["datasets"][name] = dict(id=did, dropped="no target or no numeric column")
            else:
                ok = yv != "nan"
                X, yv = X[ok], yv[ok]
                _, y = np.unique(yv, return_inverse=True)
                if len(np.unique(y)) < 2:
                    man["datasets"][name] = dict(id=did, dropped="fewer than 2 classes")
                else:
                    dest = folder / f"{name}.npz"
                    np.savez_compressed(dest, X=X, y=y.astype(np.int64))
                    man["datasets"][name] = dict(id=did, openml_name=d.details.get("name"),
                                                 version=d.details.get("version"), N=int(len(X)), F=int(X.shape[1]),
                                                 F_dropped_nonnumeric=int(d.data.shape[1] - X.shape[1]),
                                                 K=int(len(np.unique(y))), missing_cells=int(np.isnan(X).sum()),
                                                 sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        print(name, man["datasets"][name], flush=True)
        mpath.write_text(json.dumps(man, indent=1))
    lock["fetch_finished"] = dt.datetime.now().isoformat(timespec="seconds")
    (folder / "SELECTION_LOCK.json").write_text(json.dumps(lock, indent=1))
    kept = [k for k, v in man["datasets"].items() if "dropped" not in v]
    print(f"{len(kept)} of {len(sel['selected'])} datasets kept")


if __name__ == "__main__":
    main()
