"""Build and call the C kernels from Python; ELF size reader for cross-compiled objects."""
from __future__ import annotations

import ctypes as C
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .edge_model import EdgeModel

CDIR = Path(__file__).resolve().parents[1] / "c"
BUILD = CDIR / "build"


def encoder_table(m: EdgeModel) -> np.ndarray:
    """Direct encoder table ``etab[code] = #{t : tau_t <= code}`` (shared lattice, B <= 12)."""
    assert m.shared_lattice and m.B <= 12 and m.T <= 255
    return np.searchsorted(m.tau[0], np.arange(1 << m.B), side="right").astype(np.uint8)


def build_host_lib(force: bool = False) -> Path:
    BUILD.mkdir(exist_ok=True)
    lib = BUILD / "libedgehac_bench.dylib"
    srcs = [CDIR / "edgehac.c", CDIR / "bench.c"]
    if force or not lib.exists() or lib.stat().st_mtime < max(s.stat().st_mtime for s in srcs):
        subprocess.run(["clang", "-O2", "-std=c99", "-shared", "-fPIC", "-o", str(lib), str(CDIR / "bench.c")],
                       check=True)
    return lib


def _p(a, dt):
    a = np.ascontiguousarray(a, dtype=dt)
    return a, a.ctypes.data_as(C.c_void_p)


class Kernels:
    """ctypes bindings. Arrays are kept alive by the caller-visible ``self._keep`` list."""

    def __init__(self) -> None:
        self.lib = C.CDLL(str(build_host_lib()))
        for name in ("eha_lut", "eha_mac", "eha_mac_dt", "f32_nc", "i8_nc", "tree_pred"):
            getattr(self.lib, name).restype = C.c_int32
        for name in ("bench_lut", "bench_mac", "bench_mac_dt", "bench_f32", "bench_i8", "bench_tree"):
            getattr(self.lib, name).restype = C.c_double
        self._keep: list = []

    # -- helpers --------------------------------------------------------------------------------
    def _tables(self, m: EdgeModel):
        tau, tau_p = _p(m.tau[:1] if m.shared_lattice else m.tau, np.uint16)
        ts = 0 if m.shared_lattice else m.T - 1
        return tau, tau_p, ts

    def eha(self, m: EdgeModel, codes: np.ndarray, adaptive: bool = False):
        """Run the kernel over ``codes`` (N, F); returns labels, margins, cert flags (bit0 global, bit1 adaptive)."""
        N = len(codes)
        x = np.ascontiguousarray(codes, dtype=np.uint16)
        tau, tau_p, ts = self._tables(m)
        Rg, Rg_p = _p(m.R_global, np.int32)
        lab = np.empty(N, np.int32); mar = np.empty(N, np.int32); cert = np.empty(N, np.uint8)
        mg, cg = C.c_int32(), C.c_uint8()
        if m.realization == "lut":
            lut, lut_p = _p(m.lut, np.uint16)
            for i in range(N):
                lab[i] = self.lib.eha_lut(x[i].ctypes.data_as(C.c_void_p), m.K, m.F, m.T, tau_p, ts, lut_p,
                                          Rg_p, C.byref(mg), C.byref(cg))
                mar[i], cert[i] = mg.value, cg.value
        else:
            U, U_p = _p(m.U, np.uint16)
            lev, lev_p = _p(m.lev[:1] if m.shared_lattice else m.lev, np.int32)
            ls = 0 if m.shared_lattice else m.T
            beta, beta_p = _p(m.beta, np.int32)
            Dm, Dm_p = _p(m.Dmax_adapt, np.int32)
            A, A_p = _p(m.A_adapt, np.uint32)
            E, E_p = _p(m.E_adapt[:1] if m.shared_lattice else m.E_adapt, np.uint16)
            es = 0 if m.shared_lattice else m.T
            for i in range(N):
                lab[i] = self.lib.eha_mac(x[i].ctypes.data_as(C.c_void_p), m.K, m.F, m.T, tau_p, ts, U_p, lev_p, ls,
                                          beta_p, Rg_p, A_p if adaptive else None, E_p, es, Dm_p,
                                          C.byref(mg), C.byref(cg))
                mar[i], cert[i] = mg.value, cg.value
        return lab, mar, cert

    def eha_dt(self, m: EdgeModel, codes: np.ndarray):
        """Direct-table-encoder MAC kernel (shared lattice only). Returns labels, margins, global cert flags."""
        assert m.realization == "mac" and m.shared_lattice
        x = np.ascontiguousarray(codes, dtype=np.uint16)
        etab, e_p = _p(encoder_table(m), np.uint8)
        U, U_p = _p(m.U, np.uint16)
        lev, lev_p = _p(m.lev[0], np.int32)
        beta, beta_p = _p(m.beta, np.int32)
        Rg, Rg_p = _p(m.R_global, np.int32)
        N = len(x)
        lab = np.empty(N, np.int32); mar = np.empty(N, np.int32); cert = np.empty(N, np.uint8)
        mg, cg = C.c_int32(), C.c_uint8()
        for i in range(N):
            lab[i] = self.lib.eha_mac_dt(x[i].ctypes.data_as(C.c_void_p), m.K, m.F, e_p, U_p, lev_p, beta_p, Rg_p,
                                         C.byref(mg), C.byref(cg))
            mar[i], cert[i] = mg.value, cg.value
        return lab, mar, cert

    def time_eha_dt(self, m: EdgeModel, codes: np.ndarray, reps: int = 200) -> float:
        x = np.ascontiguousarray(codes, dtype=np.uint16)
        etab, e_p = _p(encoder_table(m), np.uint8)
        U, U_p = _p(m.U, np.uint16)
        lev, lev_p = _p(m.lev[0], np.int32)
        beta, beta_p = _p(m.beta, np.int32)
        Rg, Rg_p = _p(m.R_global, np.int32)
        return self.lib.bench_mac_dt(x.ctypes.data_as(C.c_void_p), len(x), reps, m.K, m.F, e_p, U_p, lev_p, beta_p, Rg_p)

    # -- timing (ns per inference, host CPU) -------------------------------------------------------
    def time_eha(self, m: EdgeModel, codes: np.ndarray, reps: int = 200, adaptive: bool = False) -> float:
        x = np.ascontiguousarray(codes, dtype=np.uint16)
        tau, tau_p, ts = self._tables(m)
        Rg, Rg_p = _p(m.R_global, np.int32)
        N = len(x)
        if m.realization == "lut":
            lut, lut_p = _p(m.lut, np.uint16)
            return self.lib.bench_lut(x.ctypes.data_as(C.c_void_p), N, reps, m.K, m.F, m.T, tau_p, ts, lut_p, Rg_p)
        U, U_p = _p(m.U, np.uint16)
        lev, lev_p = _p(m.lev[:1] if m.shared_lattice else m.lev, np.int32)
        beta, beta_p = _p(m.beta, np.int32)
        Dm, Dm_p = _p(m.Dmax_adapt, np.int32)
        A, A_p = _p(m.A_adapt, np.uint32)
        E, E_p = _p(m.E_adapt[:1] if m.shared_lattice else m.E_adapt, np.uint16)
        es = 0 if m.shared_lattice else m.T
        return self.lib.bench_mac(x.ctypes.data_as(C.c_void_p), N, reps, m.K, m.F, m.T, tau_p, ts, U_p, lev_p,
                                  0 if m.shared_lattice else m.T, beta_p, Rg_p,
                                  A_p if adaptive else None, E_p, es, Dm_p)

    def time_f32(self, codes: np.ndarray, B: int, centers: np.ndarray, w: np.ndarray | None, reps: int = 200) -> float:
        x = np.ascontiguousarray(codes, dtype=np.uint16)
        Cf, Cp = _p(centers, np.float32)
        wf, wp = _p(w if w is not None else np.ones(centers.shape[1]), np.float32)
        return self.lib.bench_f32(x.ctypes.data_as(C.c_void_p), len(x), reps, centers.shape[0], centers.shape[1],
                                  C.c_float(1.0 / ((1 << B) - 1)), Cp, wp if w is not None else None)

    def time_i8(self, x8: np.ndarray, centers_q: np.ndarray, reps: int = 200) -> float:
        x = np.ascontiguousarray(x8, dtype=np.uint8)
        Cq, Cp = _p(centers_q, np.uint8)
        return self.lib.bench_i8(x.ctypes.data_as(C.c_void_p), len(x), reps, centers_q.shape[0], centers_q.shape[1], Cp)

    @staticmethod
    def _tree_arrays(tree):
        clip0 = lambda v: np.maximum(np.asarray(v), 0)           # unused fields of leaves are -1 -> 0
        return [_p(a, dt) for a, dt in ((clip0(tree.feat), np.int16), (clip0(tree.thr), np.uint16),
                                        (clip0(tree.left), np.int16), (clip0(tree.right), np.int16),
                                        (tree.leaf, np.int16))]

    def time_tree(self, tree, codes: np.ndarray, reps: int = 200) -> float:
        x = np.ascontiguousarray(codes, dtype=np.uint16)
        arrs = self._tree_arrays(tree)
        return self.lib.bench_tree(x.ctypes.data_as(C.c_void_p), len(x), reps, x.shape[1], *[a[1] for a in arrs])

    def tree(self, tree, codes: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(codes, dtype=np.uint16)
        arrs = self._tree_arrays(tree)
        return np.array([self.lib.tree_pred(x[i].ctypes.data_as(C.c_void_p), *[a[1] for a in arrs])
                         for i in range(len(x))], dtype=np.int32)

    def i8(self, x8: np.ndarray, centers_q: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x8, dtype=np.uint8)
        Cq, Cp = _p(centers_q, np.uint8)
        mg = C.c_int32()
        return np.array([self.lib.i8_nc(x[i].ctypes.data_as(C.c_void_p), centers_q.shape[0], centers_q.shape[1], Cp,
                                        C.byref(mg)) for i in range(len(x))], dtype=np.int32)


# ------------------------------------------------------------------------------------------------
# Cross-compilation and ELF section/function sizes
# ------------------------------------------------------------------------------------------------
TARGETS = {
    "cortex-m0plus": ["--target=thumbv6m-none-eabi", "-mcpu=cortex-m0plus"],
    "cortex-m4": ["--target=thumbv7em-none-eabi", "-mcpu=cortex-m4", "-mfloat-abi=hard", "-mfpu=fpv4-sp-d16"],
    "cortex-m4-nofpu": ["--target=thumbv7em-none-eabi", "-mcpu=cortex-m4", "-mfloat-abi=soft"],
}


def _elf_functions(path: Path):
    b = path.read_bytes()
    assert b[:4] == b"\x7fELF" and b[4] == 1, "expected ELF32"
    shoff, = struct.unpack_from("<I", b, 0x20)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, 0x2E)
    secs = []
    for i in range(shnum):
        name, typ, flags, addr, off, size, link, info, align, entsize = struct.unpack_from("<10I", b, shoff + i * shentsize)
        secs.append(dict(name=name, type=typ, off=off, size=size, link=link, entsize=entsize))
    def cstr(off):
        e = b.index(b"\0", off)
        return b[off:e].decode()
    strtab_off = secs[shstrndx]["off"]
    for s in secs:
        s["sname"] = cstr(strtab_off + s["name"])
    funcs, undefined = {}, []
    for s in secs:
        if s["type"] == 2:                                             # SHT_SYMTAB
            stroff = secs[s["link"]]["off"]
            for k in range(s["size"] // 16):
                n, val, sz, info, other, shndx = struct.unpack_from("<IIIBBH", b, s["off"] + 16 * k)
                nm = cstr(stroff + n)
                if shndx == 0 and nm:
                    undefined.append(nm)
                elif (info & 0xF) == 2:
                    funcs[nm] = sz
    text = sum(s["size"] for s in secs if s["sname"].startswith(".text"))
    rodata = sum(s["size"] for s in secs if s["sname"].startswith(".rodata"))
    return funcs, undefined, text, rodata


def code_sizes(opt: str = "-Os") -> dict:
    """Per-function code size (bytes) of every kernel on each Cortex-M target + undefined runtime symbols."""
    out = {}
    with tempfile.TemporaryDirectory() as td:
        for tname, flags in TARGETS.items():
            obj = Path(td) / f"{tname}.o"
            subprocess.run(["clang", *flags, opt, "-ffreestanding", "-ffunction-sections", "-fno-builtin", "-std=c99",
                            "-c", str(CDIR / "edgehac.c"), "-o", str(obj)], check=True)
            funcs, undef, text, rodata = _elf_functions(obj)
            out[tname] = {"functions": {k: v for k, v in funcs.items() if k in
                                        ("eha_lut", "eha_mac", "eha_mac_dt", "f32_nc", "i8_nc", "tree_pred")},
                          "undefined_symbols": sorted(set(undef)), "text_total": text, "rodata": rodata}
    return out
