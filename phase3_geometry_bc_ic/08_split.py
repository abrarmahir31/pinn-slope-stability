"""Split the combined WebPlotDigitizer export into one CSV per dataset."""
import pandas as pd

SRC   = "digitised/wpd_all.csv"
NAMES = ["d_alt20","d_alt25","d_alt22","d_mkd_base","d_ground","d_mk_tm","d_f1"]
EXPECT = {"d_alt20":65,"d_alt25":15,"d_alt22":6,"d_mkd_base":49,
          "d_ground":71,"d_mk_tm":11,"d_f1":3}

raw = pd.read_csv(SRC, header=None, skiprows=2)
print(f"{raw.shape[1]} columns found, expecting {2*len(NAMES)}\n")

for i, name in enumerate(NAMES):
    pair = raw.iloc[:, 2*i:2*i+2].apply(pd.to_numeric, errors="coerce").dropna()
    pair = pair.sort_values(pair.columns[0])
    pair.to_csv(f"digitised/{name}.csv", header=False, index=False)
    flag = "" if len(pair) == EXPECT[name] else f"  <-- expected {EXPECT[name]}"
    print(f"  {name:<12} {len(pair):>3} pts   "
          f"x {pair.iloc[:,0].min():7.1f} -> {pair.iloc[:,0].max():7.1f}{flag}")
