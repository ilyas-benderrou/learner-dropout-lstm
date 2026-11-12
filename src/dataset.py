"""
Build one weekly click sequence per student enrolment from OULAD.

Output (cached in data/processed/weekly_sequences.npz):
    X      float32 [n_students, N_WEEKS, 2]
           feature 0 = total clicks in the week
           feature 1 = number of active days in the week (0..7)
    y      int     1 = at risk (Withdrawn or Fail), 0 = success (Pass or Distinction)
    unreg  float   day of unregistration (NaN if the student never unregistered)
    group  int     id_student (used to keep a student in only one split)
    outcome str    original final_result
    module str     code_module
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
CACHE = PROJECT_DIR / "data" / "processed" / "weekly_sequences.npz"
N_WEEKS = 15  # we look at the first 15 weeks of each course (~40 % of a 9-month course)
KEYS = ["code_module", "code_presentation", "id_student"]


def build_sequences(force: bool = False) -> dict[str, np.ndarray]:
    if CACHE.exists() and not force:
        data = np.load(CACHE, allow_pickle=True)
        return {k: data[k] for k in data.files}

    # ---- 1. Students, their final result and their unregistration date ----
    info = pd.read_csv(RAW_DIR / "studentInfo.csv", usecols=KEYS + ["final_result"])
    reg = pd.read_csv(RAW_DIR / "studentRegistration.csv", na_values="?")
    students = info.merge(reg, on=KEYS, how="left")

    # Students who unregistered BEFORE the course started (day <= 0) never had a
    # chance to click: predicting them is trivial and useless, so we drop them.
    before_start = students["date_unregistration"] <= 0
    print(f"Dropping {before_start.sum()} enrolments withdrawn before day 0.")
    students = students[~before_start].reset_index(drop=True)

    # Binary target. Withdrawn + Fail = the student did not complete the course
    # successfully -> this is the group a teacher would want to contact early.
    students["y"] = students["final_result"].isin(["Withdrawn", "Fail"]).astype(int)

    # ---- 2. Daily clicks -> weekly clicks (only days 0 .. 7*N_WEEKS-1) -----
    print("Reading studentVle.csv (about 10 million rows, ~30 s) ...")
    vle = pd.read_csv(
        RAW_DIR / "studentVle.csv",
        usecols=KEYS + ["date", "sum_click"],
        dtype={"code_module": "category", "code_presentation": "category",
               "id_student": "int32", "date": "int16", "sum_click": "int32"},
    )
    vle = vle[(vle["date"] >= 0) & (vle["date"] < 7 * N_WEEKS)]
    vle["week"] = (vle["date"] // 7).astype("int8")

    # clicks per day first (a student can click several resources on one day)
    daily = vle.groupby(KEYS + ["week", "date"], observed=True)["sum_click"].sum().reset_index()
    weekly = daily.groupby(KEYS + ["week"], observed=True).agg(
        clicks=("sum_click", "sum"), active_days=("date", "nunique")).reset_index()
    for k in ["code_module", "code_presentation"]:
        weekly[k] = weekly[k].astype(str)

    # ---- 3. Fill a dense array [student, week, feature] (missing week = 0) ----
    index = {tuple(r): i for i, r in enumerate(students[KEYS].itertuples(index=False))}
    X = np.zeros((len(students), N_WEEKS, 2), dtype=np.float32)
    rows = weekly[KEYS].apply(tuple, axis=1).map(index)
    ok = rows.notna()
    r = rows[ok].astype(int).to_numpy()
    w = weekly.loc[ok, "week"].to_numpy()
    X[r, w, 0] = weekly.loc[ok, "clicks"].to_numpy()
    X[r, w, 1] = weekly.loc[ok, "active_days"].to_numpy()

    data = {
        "X": X,
        "y": students["y"].to_numpy(),
        "unreg": students["date_unregistration"].to_numpy(dtype=float),
        "group": students["id_student"].to_numpy(),
        "outcome": students["final_result"].to_numpy(dtype=str),
        "module": students["code_module"].to_numpy(dtype=str),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **data)
    print(f"Saved {len(students)} sequences to {CACHE}")
    return data


if __name__ == "__main__":
    d = build_sequences(force=True)
    print(d["X"].shape, "at-risk ratio:", d["y"].mean().round(3))
