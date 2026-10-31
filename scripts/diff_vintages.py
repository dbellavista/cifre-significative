"""Show what changed between the two most recent snapshots of a table (or two given files).

Usage: uv run python scripts/diff_vintages.py [panel|regional|regional_extra|highfreq|all] [old.parquet new.parquet]
"""
import sys
from pathlib import Path

import pandas as pd

from cifresignificative.core.clean.vintages import TABLES, VINTAGE_DIR, diff

args = sys.argv[1:]
name = args.pop(0) if args and args[0] in {*TABLES, "all"} else "panel"
if name == "all" and args:
    raise SystemExit("`all` takes no file arguments")
for tab in TABLES if name == "all" else [name]:
    files = [Path(a) for a in args[:2]] or sorted(VINTAGE_DIR.glob(f"{tab}_2*.parquet"))[-2:]
    if len(files) < 2:
        print(f"{tab}: need at least two snapshots (run `cifresignificative.core.clean.build` after two different data pulls)")
        continue
    old, new = (pd.read_parquet(f) for f in files)
    res = diff(old, new, key=TABLES[tab][0])
    print(f"\n== {tab}: {files[0].name} -> {files[1].name}")
    for k, d in res.items():
        print(f"\n{k}: {len(d)} points" + (f" in {d.indicator.nunique()} indicators" if len(d) else ""))
        if len(d):
            print(d.indicator.value_counts().head(10).to_string())
    if len(res["revised"]):
        print("\nlargest revisions:")
        print(res["revised"].head(10).round(2).to_string(index=False))
