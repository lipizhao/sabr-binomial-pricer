"""Percentile averages of single-name stress shocks from SingleNames.xlsx.

Sheet layout expected:
    row 1 : name labels (SN1, SN2, ...) above the first column of each block
    row 2 : 'dates', then per name: 0.8, 0.9, 1, 1.1, 1.2, '80-120'
    row 3+: data; column A holds the dates shared by all names
"""
from pathlib import Path

import numpy as np
import pandas as pd

SHOCK_COLS = ["80%", "90%", "100%", "110%", "120%"]
SORT_COL = "80-120"
OUT_COLS = SHOCK_COLS + [SORT_COL]


def _header_label(h) -> str:
    """Map header cell (0.8, 1, '80-120', ...) to the output column name."""
    if isinstance(h, str):
        return h.strip()
    return f"{round(float(h) * 100)}%"


def split_by_name(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Step 1: break the wide sheet into one table per name (based on row 1)."""
    names_row, header_row, data = raw.iloc[0], raw.iloc[1], raw.iloc[2:]
    dates = pd.to_datetime(data.iloc[:, 0], errors="coerce")

    name_cols = [c for c in raw.columns[1:] if pd.notna(names_row[c])]
    tables = {}
    for i, start in enumerate(name_cols):
        end = name_cols[i + 1] if i + 1 < len(name_cols) else raw.columns[-1] + 1
        block_cols = [c for c in raw.columns if start <= c < end]
        block = data[block_cols].copy()
        block.columns = [_header_label(header_row[c]) for c in block_cols]
        block = block[OUT_COLS].apply(pd.to_numeric, errors="coerce")
        block.insert(0, "date", dates.values)
        block = block.dropna(subset=OUT_COLS, how="all")
        tables[str(names_row[start]).strip()] = block.reset_index(drop=True)
    return tables


def stack_and_sort(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Step 2: stack all names vertically, sort by '80-120' from large to small."""
    stacked = pd.concat(
        [t.assign(name=n) for n, t in tables.items()], ignore_index=True
    )
    stacked = stacked[["name", "date"] + OUT_COLS]
    stacked = stacked.dropna(subset=[SORT_COL])
    return stacked.sort_values(SORT_COL, ascending=False, ignore_index=True)


def percentile_row(n: int, p: float) -> int:
    """0-based row k in a descending-sorted table for percentile p.

    With descending order, the p-th percentile sits a fraction (1 - p) from
    the top, so 99% is near the top and 1% is near the bottom. k is clamped
    to [1, n-2] so rows k-1 and k+1 always exist.
    """
    if n < 3:
        raise ValueError(f"Need at least 3 rows, got {n}")
    k = int(round((1.0 - p) * (n - 1)))
    return min(max(k, 1), n - 2)


def percentile_averages(
    sorted_df: pd.DataFrame, percentiles=(0.99, 0.975, 0.025, 0.01)
) -> pd.DataFrame:
    """Steps 3-5: average rows k-1, k, k+1 of the shock columns at each percentile."""
    n = len(sorted_df)
    rows = []
    for p in percentiles:
        k = percentile_row(n, p)
        window = sorted_df.iloc[k - 1 : k + 2]
        rec = {"percentile": f"{p:.1%}", "k": k, f"{SORT_COL} at k": sorted_df[SORT_COL].iat[k]}
        rec.update(window[SHOCK_COLS].mean().to_dict())
        rows.append(rec)
    return pd.DataFrame(rows).set_index("percentile")


def stress_percentiles(xlsx_path, sheet_name="SingleNames", percentiles=(0.99, 0.975, 0.025, 0.01)):
    """Read the sheet and return (sorted stacked table, percentile averages)."""
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    raw.columns = range(raw.shape[1])
    sorted_df = stack_and_sort(split_by_name(raw))
    return sorted_df, percentile_averages(sorted_df, percentiles)


if __name__ == "__main__":
    path = Path(__file__).with_name("SingleNames.xlsx")
    sorted_df, result = stress_percentiles(path)
    pd.set_option("display.width", 200)
    print(sorted_df.to_string())
    print()
    print(result.round(4).to_string())
