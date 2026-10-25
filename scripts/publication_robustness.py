"""Robustness of the synthetic Italy of post 00: other donor pools and fitting windows.

Usage: uv run python scripts/publication_robustness.py   -> one line per variant (gap in log points x100, post/pre RMSPE ratio and Italy's rank among the placebo units).
"""
import warnings

import numpy as np

from cifresignificative.analysis.data import ITALY, load_long, long_wide
from cifresignificative.analysis.synth import placebo_ratios, rmspe, synthetic_control

warnings.filterwarnings("ignore")
END = 2025
lh = load_long(); gdp = long_wide(lh, "gdp_real_per_head_ncu")
y = (np.log(gdp) - np.log(gdp.loc[1995])) * 100
ALL = [c for c in y.columns if c != "EU27" and y.loc[1970:END, c].notna().all()]  # every country of the panel with a complete series from 1970
def run(label, drop=(), pre=(1970,1994)):
    units = [c for c in ALL if c not in drop]; donors = [c for c in units if c != ITALY]
    yy = y.loc[1970:END, units]
    w, s = synthetic_control(yy, ITALY, pre, donors); gap = yy[ITALY]-s
    r = placebo_ratios(yy, units, pre, (1995, END))
    top = {k: round(v,2) for k,v in w[w>0.05].sort_values(ascending=False).items()}
    print(f"{label:28s} n={len(units):2d} preRMSPE {rmspe(yy[ITALY],s,pre):4.1f} gap05 {gap.loc[2005]:6.1f} gap25 {gap.loc[END]:6.1f} ratio {r[ITALY]:5.1f} rank {int(r.rank(ascending=False)[ITALY])}/{len(r)} 2nd {r.drop(ITALY).max():.1f} {r.drop(ITALY).idxmax()} w {top}")
run("baseline")
run("no JPN", ["JPN"])
run("no JPN,USA,GBR (EU only)", ["JPN","USA","GBR"])
run("no FRA", ["FRA"])
run("no FRA,JPN", ["FRA","JPN"])
run("no FRA,AUT,JPN", ["FRA","AUT","JPN"])
run("no euro-area-outsiders+DEU", ["JPN","USA","GBR","DNK","SWE"])
run("pre 1980-1994", pre=(1980,1994))
run("pre 1970-1989", pre=(1970,1989))
run("pre 1970-1999", pre=(1970,1999))
