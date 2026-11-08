"""Install the ExKMC library (IMM / ExKMC threshold-tree baselines) exactly as used in the study.

ExKMC 0.0.3 (Frost, Moshkovitz and Rashtchian, MIT licence) is distributed on PyPI as a source package whose splitters
are a Cython extension; that extension does not build on recent Python versions. The study therefore used the package's
own pure-Python splitters (ExKMC/splitters/py_cut_finder.py, shipped in the same source package): the only change is
that ExKMC/splitters/__init__.py imports from .py_cut_finder instead of the compiled cut_finder module.

This script downloads the source package from PyPI, checks its SHA-256, applies that two-line change and copies the
package into the site-packages of the running interpreter:

    .venv/bin/python tools/install_exkmc.py
"""
import hashlib
import io
import shutil
import sysconfig
import tarfile
import urllib.request
from pathlib import Path

URL = ("https://files.pythonhosted.org/packages/47/a9/e3870f54a6b7f44744e77c7fef50daae174292c8c477b3d186e0579b11cd/"
       "ExKMC-0.0.3.tar.gz")
SHA256 = "97b1cd7bad2dff36b855b21943526592b2ff5f8ffd181d9f4da3f3bac3295c10"
PATCHED_INIT = ("from .py_cut_finder import get_min_mistakes_cut\n"
                "from .py_cut_finder import get_min_surrogate_cut\n")


def main() -> None:
    dest = Path(sysconfig.get_paths()["purelib"]) / "ExKMC"
    data = urllib.request.urlopen(URL, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise SystemExit("ExKMC-0.0.3.tar.gz: SHA-256 mismatch, not installed")
    if dest.exists():
        shutil.rmtree(dest)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for m in tar.getmembers():
            parts = Path(m.name).parts
            if len(parts) < 3 or parts[1] != "ExKMC" or not m.isfile():
                continue
            out = dest.joinpath(*parts[2:])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(tar.extractfile(m).read())
    init = dest / "splitters" / "__init__.py"
    init.write_text(init.read_text().replace("from cut_finder import", "from .py_cut_finder import"))
    assert init.read_text().strip() == PATCHED_INIT.strip(), "unexpected ExKMC splitters/__init__.py"
    print(f"installed ExKMC 0.0.3 (pure-Python splitters) into {dest}")


if __name__ == "__main__":
    main()
