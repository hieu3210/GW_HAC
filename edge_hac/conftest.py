import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):          # edge_hac/ (edgehac) and the repo root (haclustering core)
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
