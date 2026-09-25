"""Load the Hu et al. 2025 experimental Slide-recon data (SCP2577) into the
same (C, X_true, U_true) form as common.simulate().

Replicates the preprocessing in analysis_scripts/Real_Data_Exploratory.ipynb:
bead-bead read counts -> anchor(V10, receiver) x target(V9A30, sender) matrix,
matched to in-situ-sequencing ground-truth positions (pixel -> um, *0.72),
filtered to beads with a known position and max count > 10.

Files live on `bio` under ~/diff_seq/SCP2577/.  Four replicates available:
c58_8 (the one used in the paper), c58_4, c58_5, c58_16.
"""
import os
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
import editdistance

DIFF_SEQ = os.path.expanduser("~/diff_seq")
PIXEL_TO_UM = 0.72
ANCHOR, TARGET = "V10", "V9A30"          # receiver, sender

REPS = {
    "c58_8":  ("SCP2577/expression/c58_8_blind_raw_reads_filtered.csv.gz",
               "SCP2577/documentation/c58_8_matched_bead_location.csv.gz"),
    "c58_4":  ("SCP2577/documentation/c58_4_blind_raw_reads_filtered.csv.gz",
               "SCP2577/documentation/c58_4_matched_bead_location.csv.gz"),
    "c58_5":  ("SCP2577/documentation/c58_5_blind_raw_reads_filtered.csv.gz",
               "SCP2577/documentation/c58_5_matched_bead_location.csv.gz"),
    "c58_16": ("SCP2577/documentation/c58_16_blind_raw_reads_filtered.csv.gz",
               "SCP2577/documentation/c58_16_matched_bead_location.csv.gz"),
}


def _get_matrix(match_df, min_a, max_a, min_t, max_t, anchor, target):
    a_all = match_df.groupby(anchor)["cnt"].sum().reset_index(name="total_cnt")
    a_sel = a_all.loc[(a_all.total_cnt > min_a) & (a_all.total_cnt < max_a)]
    t_all = match_df.groupby(target)["cnt"].sum().reset_index(name="total_cnt")
    t_sel = t_all.loc[(t_all.total_cnt > min_t) & (t_all.total_cnt < max_t)]
    match_df = match_df[match_df[anchor].isin(a_sel[anchor])
                        & match_df[target].isin(t_sel[target])]
    a_list = match_df.groupby(anchor)["cnt"].sum().reset_index(name="total_cnt")
    t_list = match_df.groupby(target)["cnt"].sum().reset_index(name="total_cnt")
    a_dict = {a_list.iloc[i, 0]: i for i in range(len(a_list))}
    t_dict = {t_list.iloc[j, 0]: j for j in range(len(t_list))}
    a_coo = [a_dict[a] for a in match_df[anchor]]
    t_coo = [t_dict[t] for t in match_df[target]]
    counts = sp.coo_matrix((match_df["cnt"], (a_coo, t_coo))).tocsr()
    return counts, a_list, t_list


def _build_6mer(bc_list):
    s, m, e = {}, {}, {}
    for bc in bc_list:
        s.setdefault(bc[:6], []).append(bc)
        m.setdefault(bc[4:10], []).append(bc)
        e.setdefault(bc[-6:], []).append(bc)
    return s, m, e


def _barcode_matching(bc_pos_dict, spatial_bc_list, max_dist=1):
    out = {}
    ref = list(bc_pos_dict.keys())
    start_km, mid_km, end_km = _build_6mer(ref)

    def cand(bc):
        r = []
        if bc[:6] in start_km:
            r += start_km[bc[:6]]
        if bc[-6:] in end_km:
            r += end_km[bc[-6:]]
        if bc[4:10] in mid_km:
            r += mid_km[bc[4:10]]
        return set(r)

    for bc in spatial_bc_list:
        if bc in bc_pos_dict:
            out[bc] = bc
            continue
        fz = [(it, editdistance.eval(it, bc)) for it in cand(bc)]
        fz = sorted((it for it in fz if it[1] <= max_dist), key=lambda x: x[1])
        if fz:
            out[bc] = fz[0][0]
    return out


def load_hu(rep="c58_8", min_max_count=10):
    reads_path, pos_path = (os.path.join(DIFF_SEQ, p) for p in REPS[rep])

    blind = pd.read_csv(reads_path, compression="gzip")
    bs = (blind.groupby(["R1_bc", "R2_bc"])["reads"].sum().reset_index())
    bs.columns = [ANCHOR, TARGET, "cnt"]

    pos = pd.read_csv(pos_path, compression="gzip")
    pos["xcoord"] = (pos["xcoord"] - pos["xcoord"].mean()) * PIXEL_TO_UM
    pos["ycoord"] = (pos["ycoord"] - pos["ycoord"].mean()) * PIXEL_TO_UM

    counts_temp, a_temp, t_temp = _get_matrix(bs, 0, 100000, 0, 100000, ANCHOR, TARGET)

    a_match = _barcode_matching(Counter(pos["barcode"]), a_temp[ANCHOR].values)
    t_match = _barcode_matching(Counter(pos["barcode"]), t_temp[TARGET].values)
    a_keep = set(pd.Series(list(a_match.values())).unique())
    t_keep = set(pd.Series(list(t_match.values())).unique())

    a_mask = a_temp[ANCHOR].isin(a_keep).values
    t_mask = t_temp[TARGET].isin(t_keep).values
    counts = counts_temp.toarray()[np.ix_(a_mask, t_mask)]

    a_sel = a_temp[a_mask].merge(pos, how="left", left_on=ANCHOR, right_on="barcode")
    t_sel = t_temp[t_mask].merge(pos, how="left", left_on=TARGET, right_on="barcode")

    ok_a = a_sel["xcoord"].notna().values
    ok_t = t_sel["xcoord"].notna().values
    counts = counts[np.ix_(ok_a, ok_t)]
    Xr = a_sel.loc[ok_a, ["xcoord", "ycoord"]].to_numpy(float)
    Us = t_sel.loc[ok_t, ["xcoord", "ycoord"]].to_numpy(float)

    keep_r = counts.max(axis=1) > min_max_count
    keep_s = counts.max(axis=0) > min_max_count
    counts = counts[np.ix_(keep_r, keep_s)]
    Xr, Us = Xr[keep_r], Us[keep_s]

    return counts.astype(np.float64), Xr, Us


if __name__ == "__main__":
    import sys
    rep = sys.argv[1] if len(sys.argv) > 1 else "c58_8"
    C, Xr, Us = load_hu(rep)
    print(f"{rep}: C {C.shape}  receiver UMI/bead {C.sum(1).mean():.0f}  "
          f"field ~{(Xr.max(0) - Xr.min(0)).mean():.0f} um")
