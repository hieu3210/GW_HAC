"""Resource model: exact operation / memory counts and an *illustrative* Cortex-M4 cycle estimate.

Counts are exact for the kernels in ``c/edgehac.c``.  The cycle numbers are NOT measurements: they apply the
per-instruction cycle table below (Cortex-M4 technical reference manual, zero-wait-state memory, no
compiler scheduling beyond the naive instruction sequence) and serve only to rank approaches.  No on-device
timing was performed in this study.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

# cycles per primitive on Cortex-M4(F)   (LDR 2, ALU 1, MLA 1, branch ~2 average, VFP: VLDR 2, VSUB 1, VMLA 3)
CYC = dict(load=2, alu=1, mac=1, branch=2, fld=2, fsub=1, fmla=3, fmul=1, fcvt=1, loop=1)


@dataclass
class Footprint:
    method: str
    flash_bytes: int            # constant model tables
    ram_bytes: int              # working buffers (indices, scores)
    ops: dict                   # primitive counts per inference
    cycles_m4: float            # illustrative estimate

    def row(self) -> dict:
        d = asdict(self)
        d["ops"] = "; ".join(f"{k}={v:g}" for k, v in self.ops.items())
        return d


def _bsearch_steps(T: int) -> int:
    return int(np.ceil(np.log2(T)))          # comparisons per feature in the binary search over T-1 thresholds


def edge_footprint(model, adaptive: bool = False, direct_encoder: bool = False) -> Footprint:
    K, F, T = model.K, model.F, model.T
    steps = _bsearch_steps(T)
    enc_cyc = F * steps * (CYC["load"] + CYC["alu"] + CYC["branch"] + CYC["loop"])
    if direct_encoder:                       # one byte load per feature (code is already in a register-width load)
        enc_cyc = F * (CYC["load"] + CYC["load"])
    mem = model.memory_bytes(adaptive)
    if model.realization == "lut":
        pair = CYC["load"] + CYC["alu"] + CYC["loop"] + 1            # lut read, add, loop, address update
        ops = dict(compare=F * steps, table_read=K * F, add=K * F, multiply=0)
        name = "Edge-HAC/LUT"
        ram = F * 2 + K * 4
    else:
        pair = CYC["load"] + CYC["mac"] + CYC["loop"] + 1             # U read, MLA, loop, address update
        ops = dict(compare=F * steps, table_read=F, mac=K * F)
        name = "Edge-HAC/MAC"
        ram = F * 2 + K * 4
    dist_cyc = K * F * pair
    argmin_cyc = K * (CYC["alu"] + CYC["branch"])
    cert_cyc = CYC["load"] + CYC["alu"] + CYC["branch"]
    cyc = enc_cyc + dist_cyc + argmin_cyc + cert_cyc
    if direct_encoder:
        mem_total = mem["total"] + (1 << model.B)
        ops = {**ops, "compare": 0, "encoder_table_read": F}
        name += "+direct-encoder"
    else:
        mem_total = mem["total"]
    if adaptive:
        cyc += F * (CYC["load"] * 2 + 2 + CYC["loop"])                  # A, E reads, UMLAL, loop
        ops["mac_adaptive_u64"] = F
        name += "+adaptive-cert"
    return Footprint(name, mem_total, ram, ops, float(cyc))


def float_footprint(K: int, F: int, weighted: bool = False) -> Footprint:
    pair = CYC["fld"] + CYC["fsub"] + CYC["fmla"] + CYC["loop"] + (CYC["fmul"] if weighted else 0)
    cyc = F * (CYC["fcvt"] + CYC["fmul"]) + K * F * pair + K * (CYC["fsub"] + CYC["branch"])
    flash = 4 * K * F + (4 * F if weighted else 0)
    return Footprint("float32 EWA-HAC" if weighted else "float32 k-means", flash, 4 * F + 4 * K,
                     dict(fsub=K * F, fmla=K * F, fcvt=F), float(cyc))


def int8_footprint(K: int, F: int) -> Footprint:
    pair = CYC["load"] + CYC["alu"] + CYC["mac"] + CYC["loop"]
    cyc = K * F * pair + K * (CYC["alu"] + CYC["branch"])
    return Footprint("int8-PTQ k-means", K * F, F + 4 * K, dict(sub=K * F, mac=K * F), float(cyc))


def tree_footprint(tree, avg_steps: float) -> Footprint:
    per_level = CYC["load"] * 3 + CYC["alu"] + CYC["branch"]
    return Footprint("IMM tree", tree.memory_bytes(), 4, dict(compare=avg_steps),
                     float(avg_steps * per_level))
