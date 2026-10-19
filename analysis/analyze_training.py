"""Exploratory analysis of the TrackID3x3 (ABC2027) official Training set.

Usage:
    python analysis/analyze_training.py DATA_DIR [OUT_DIR]

DATA_DIR must contain (from the official challenge Drive folder):
    bbox/IMG_01xx.txt     Training/full player annotations/Image-space bounding boxes
    court/IMG_01xx.txt    Training/full player annotations/court position
    train_court_registration.csv, train_games_roster.csv
    final_court_registration.csv, final_games_roster.csv
Only official Training data is used (no public GT of the Challenge sequences).
"""
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

FPS = 30000 / 1001
COURT_W, COURT_H = 1505.0, 1105.0
TAU = 100.0
ALPHAS = np.arange(0.05, 1.0, 0.05)
VIDEOS = [f"IMG_01{i:02d}" for i in range(4, 12)]


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------

def homography(src, dst):
    a = []
    for (x, y), (u, v) in zip(src, dst):
        a += [[-x, -y, -1, 0, 0, 0, u * x, u * y, u],
              [0, 0, 0, -x, -y, -1, v * x, v * y, v]]
    _, _, vt = np.linalg.svd(np.asarray(a, float))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


def apply_h(h, x, y):
    p = h @ np.vstack([x, y, np.ones_like(x)])
    return p[0] / p[2], p[1] / p[2]


def camera_from_homography(h_court2img, img_w, img_h):
    """Recover K (square pixels, centred principal point), R, t from a plane homography."""
    cx, cy = img_w / 2, img_h / 2
    shift = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    g = shift @ h_court2img
    h1, h2 = g[:, 0], g[:, 1]
    # r1 . r2 = 0 with K = diag(f, f, 1)
    f2 = -(h1[0] * h2[0] + h1[1] * h2[1]) / (h1[2] * h2[2])
    f = math.sqrt(f2)
    k = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    m = np.linalg.inv(k) @ h_court2img
    lam = 1 / np.linalg.norm(m[:, 0])
    r1, r2, t = m[:, 0] * lam, m[:, 1] * lam, m[:, 2] * lam
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    return k, np.column_stack([r1, r2, r3]), t


def project3d(k, r, t, xyz):
    p = k @ (r @ xyz.T + t[:, None])
    return p[0] / p[2], p[1] / p[2]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load(data_dir):
    reg = pd.read_csv(f"{data_dir}/train_court_registration.csv")
    h_i2c = homography(reg[["image_x", "image_y"]].values, reg[["court_x", "court_y"]].values)
    roster = pd.read_csv(f"{data_dir}/train_games_roster.csv")
    frames = []
    stats = {}
    for v in VIDEOS:
        b = pd.read_csv(f"{data_dir}/bbox/{v}.txt", header=None).iloc[:, :6]
        b.columns = ["f", "tid", "x", "y", "w", "h"]
        c = pd.read_csv(f"{data_dir}/court/{v}.txt", header=None,
                        names=["f", "cx", "cy", "team", "num"])
        b["px"], b["py"] = apply_h(h_i2c, (b.x + b.w / 2).values, (b.y + b.h).values)
        # Join bbox rows to identity rows per frame through the exact projection.
        bg = dict(tuple(b.groupby("f")))
        joined, errs = [], []
        for f, g in c.groupby("f"):
            bb = bg.get(f)
            if bb is None:
                continue
            d = np.hypot(g.cx.values[:, None] - bb.px.values[None], g.cy.values[:, None] - bb.py.values[None])
            ri, ci = linear_sum_assignment(d)
            errs += list(d[ri, ci])
            jj = g.iloc[ri].reset_index(drop=True)
            jj[["tid", "x", "y", "w", "h"]] = bb.iloc[ci][["tid", "x", "y", "w", "h"]].values
            joined.append(jj)
        d = pd.concat(joined, ignore_index=True)
        d["video"] = v
        d["ident"] = d.team + d.num.astype(str)
        gs = int(roster.loc[roster.video == v, "game_start_frame"].iloc[0])
        d["pregame"] = d.f < gs
        per = c.groupby("f").size()
        dup = int((c.groupby(["f", "team", "num"]).size() > 1).sum())
        stats[v] = dict(frames=int(c.f.max()), game_start=gs, pregame_sec=round((gs - 1) / FPS, 1),
                        duration_min=round(c.f.max() / FPS / 60, 2),
                        rows_per_frame={int(k): int(n) for k, n in per.value_counts().items()},
                        duplicate_identity_rows=dup,
                        bbox_rows=len(b), court_rows=len(c),
                        proj_err_max_cm=float(np.max(errs)))
        frames.append(d)
    return pd.concat(frames, ignore_index=True), stats, h_i2c


# ----------------------------------------------------------------------------
# TI-HOTA (same logic as the official notebook, vectorised over alphas)
# ----------------------------------------------------------------------------

def ti_hota(gt, pred, tau=TAU):
    """gt/pred: DataFrames with f, cx, cy, ident. Returns dict like the official code."""
    gtg, prg = dict(tuple(gt.groupby("f"))), dict(tuple(pred.groupby("f")))
    na = len(ALPHAS)
    tp, fp, fn = np.zeros(na), np.zeros(na), np.zeros(na)
    tpa, fna, fpa = [dict() for _ in ALPHAS], [dict() for _ in ALPHAS], [dict() for _ in ALPHAS]
    for f in sorted(set(gtg) | set(prg)):
        g, p = gtg.get(f), prg.get(f)
        gi = [] if g is None else list(g.ident)
        pi = [] if p is None else list(p.ident)
        if g is not None and p is not None:
            d2 = (g.cx.values[:, None] - p.cx.values[None]) ** 2 + (g.cy.values[:, None] - p.cy.values[None]) ** 2
            sim = np.exp(math.log(0.05) * d2 / tau ** 2) * (np.array(gi)[:, None] == np.array(pi)[None])
        else:
            sim = np.zeros((len(gi), len(pi)))
        for a_i, a in enumerate(ALPHAS):
            match = {}
            if sim.size:
                cost = 1 - sim
                cost[sim < a] = 1e6
                r, c = linear_sum_assignment(cost)
                match = {i: j for i, j in zip(r, c) if sim[i, j] >= a}
            tp[a_i] += len(match)
            fp[a_i] += len(pi) - len(match)
            fn[a_i] += len(gi) - len(match)
            for i, gid in enumerate(gi):
                key = "tpa" if i in match else "fna"
                dct = tpa[a_i] if key == "tpa" else fna[a_i]
                dct[gid] = dct.get(gid, 0) + 1
            mv = set(match.values())
            for j, pid in enumerate(pi):
                if j not in mv:
                    fpa[a_i][pid] = fpa[a_i].get(pid, 0) + 1
    det, ass = [], []
    for a_i in range(na):
        det.append(tp[a_i] / max(tp[a_i] + fp[a_i] + fn[a_i], 1))
        ids = set(tpa[a_i]) | set(fna[a_i])
        vals = [tpa[a_i].get(i, 0) / (tpa[a_i].get(i, 0) + fna[a_i].get(i, 0) + fpa[a_i].get(i, 0)) for i in ids]
        ass.append(np.mean(vals) if vals else 0)
    det, ass = np.array(det), np.array(ass)
    return {"TI-HOTA": float(np.mean(np.sqrt(det * ass))), "TI-DetA": float(det.mean()),
            "TI-AssA": float(ass.mean())}


# ----------------------------------------------------------------------------
# Analyses
# ----------------------------------------------------------------------------

def court_lines(ax):
    ax.add_patch(plt.Rectangle((0, 0), COURT_W, COURT_H, fill=False, color="k", lw=1))
    ax.set_xlim(-300, COURT_W + 300)
    ax.set_ylim(COURT_H + 300, -500)
    ax.set_aspect("equal")


def kinematics(d):
    out = []
    for (v, ident), g in d.sort_values("f").groupby(["video", "ident"]):
        f = g.f.values
        ok = np.diff(f) == 1
        dx, dy = np.diff(g.cx.values), np.diff(g.cy.values)
        sp = np.hypot(dx, dy)[ok] * FPS / 100  # m/s
        out.append(pd.DataFrame({"video": v, "ident": ident, "speed": sp, "pregame": g.pregame.values[1:][ok]}))
    return pd.concat(out, ignore_index=True)


def gt_jitter(d):
    """Residual between GT trajectory and a Savitzky-Golay smoothed version (annotation noise)."""
    res = []
    for _, g in d.sort_values("f").groupby(["video", "ident"]):
        if len(g) < 31 or (np.diff(g.f.values) != 1).any():
            continue
        for col in ("cx", "cy"):
            s = savgol_filter(g[col].values, 15, 2)
            res.append(g[col].values - s)
    r = np.concatenate(res)
    return float(np.std(r)), float(np.percentile(np.abs(r), 95))


def proximity(d):
    rows = []
    for (v, f), g in d.groupby(["video", "f"]):
        if len(g) < 2:
            continue
        xy, t = g[["cx", "cy"]].values, g.team.values
        dist = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        iu = np.triu_indices(len(g), 1)
        same = t[iu[0]] == t[iu[1]]
        dd = dist[iu]
        rows.append((v, f, dd.min(), dd[same].min() if same.any() else np.nan,
                     dd[~same].min() if (~same).any() else np.nan, bool(g.pregame.iloc[0])))
    return pd.DataFrame(rows, columns=["video", "f", "min_any", "min_same", "min_opp", "pregame"])


def bbox_overlap(d):
    """Fraction of player boxes overlapping another box (IoU>0, >0.3) in the training view."""
    n0 = n3 = tot = 0
    for _, g in d.groupby(["video", "f"]):
        x1, y1 = g.x.values, g.y.values
        x2, y2 = x1 + g.w.values, y1 + g.h.values
        iw = np.clip(np.minimum(x2[:, None], x2[None]) - np.maximum(x1[:, None], x1[None]), 0, None)
        ih = np.clip(np.minimum(y2[:, None], y2[None]) - np.maximum(y1[:, None], y1[None]), 0, None)
        inter = iw * ih
        area = (x2 - x1) * (y2 - y1)
        iou = inter / (area[:, None] + area[None] - inter)
        np.fill_diagonal(iou, 0)
        m = iou.max(1)
        n0 += (m > 0).sum()
        n3 += (m > 0.3).sum()
        tot += len(g)
    return n0 / tot, n3 / tot


def synth_boxes(k, r, t, cx, cy, height=172.0, width_ratio=0.38):
    fx, fy = project3d(k, r, t, np.column_stack([cx, cy, np.zeros_like(cx)]))
    hx, hy = project3d(k, r, t, np.column_stack([cx, cy, np.full_like(cx, height)]))
    hpx = np.abs(fy - hy)
    return fx, fy, hpx, hpx * width_ratio


def final_view_overlap(d, cam, height=172.0):
    k, r, t = cam
    n3 = tot = 0
    hp = []
    for _, g in d[~d.pregame].groupby(["video", "f"]):
        fx, fy, hpx, wpx = synth_boxes(k, r, t, g.cx.values, g.cy.values, height)
        hp += list(hpx)
        x1, x2, y1, y2 = fx - wpx / 2, fx + wpx / 2, fy - hpx, fy
        iw = np.clip(np.minimum(x2[:, None], x2[None]) - np.maximum(x1[:, None], x1[None]), 0, None)
        ih = np.clip(np.minimum(y2[:, None], y2[None]) - np.maximum(y1[:, None], y1[None]), 0, None)
        inter = iw * ih
        area = (x2 - x1) * (y2 - y1)
        iou = inter / (area[:, None] + area[None] - inter)
        np.fill_diagonal(iou, 0)
        n3 += (iou.max(1) > 0.3).sum()
        tot += len(g)
    return n3 / tot, np.array(hp)


def sensitivity(d, h_final_c2i, video="IMG_0104", seed=0):
    rng = np.random.default_rng(seed)
    gt = d[d.video == video][["f", "cx", "cy", "ident", "team", "num"]].reset_index(drop=True)
    out = {"court_noise_cm": {}, "final_view_pixel_noise": {}, "drop_rate": {}, "id_swap_rate": {},
           "team_flip": None}
    for s in [0, 5, 10, 15, 20, 30, 50]:
        p = gt.copy()
        p["cx"] += rng.normal(0, s, len(p))
        p["cy"] += rng.normal(0, s, len(p))
        out["court_noise_cm"][s] = ti_hota(gt, p)["TI-HOTA"]
    h_i2c = np.linalg.inv(h_final_c2i)
    u, v = apply_h(h_final_c2i, gt.cx.values, gt.cy.values)
    for s in [0, 1, 2, 3, 5, 8]:
        p = gt.copy()
        p["cx"], p["cy"] = apply_h(h_i2c, u + rng.normal(0, s, len(p)), v + rng.normal(0, s, len(p)))
        out["final_view_pixel_noise"][s] = ti_hota(gt, p)["TI-HOTA"]
    for r_ in [0.05, 0.1, 0.2, 0.3]:
        p = gt[rng.random(len(gt)) > r_]
        out["drop_rate"][r_] = ti_hota(gt, p)["TI-HOTA"]
    # Identity swap: in a fraction of frames, swap two same-team identities.
    frames = gt.f.unique()
    for r_ in [0.05, 0.1, 0.2]:
        p = gt.copy()
        bad = set(rng.choice(frames, int(r_ * len(frames)), replace=False))
        idx = p[p.f.isin(bad) & (p.team == "A")].groupby("f").head(2).index.values.reshape(-1, 2)
        a, b = idx[:, 0], idx[:, 1]
        p.loc[a, "ident"], p.loc[b, "ident"] = gt.loc[b, "ident"].values, gt.loc[a, "ident"].values
        out["id_swap_rate"][r_] = ti_hota(gt, p)["TI-HOTA"]
    p = gt.copy()
    p["ident"] = np.where(p.team == "A", "B", "A") + p.num.astype(str)
    out["team_flip"] = ti_hota(gt, p)["TI-HOTA"]
    # Predicting wrong-but-close vs omitting: one identity always 60 cm off.
    p = gt.copy()
    m = p.ident == p.ident.iloc[0]
    p.loc[m, "cx"] += 60
    out["one_player_60cm_off"] = ti_hota(gt, p)["TI-HOTA"]
    out["one_player_omitted"] = ti_hota(gt, gt[~m])["TI-HOTA"]
    return out


def gt_outliers(d, cam_pos_xy):
    """Deviation of GT from a 9-frame median, split into radial (along the training-camera ray) and tangential."""
    from scipy.ndimage import median_filter
    out = {}
    for v, g in d.groupby("video"):
        p = g.pivot_table(index="f", columns="ident", values=["cx", "cy"])
        x, y = p["cx"].values, p["cy"].values
        xm = median_filter(np.nan_to_num(x), size=(9, 1))
        ym = median_filter(np.nan_to_num(y), size=(9, 1))
        dx, dy = x - xm, y - ym
        rx, ry = xm - cam_pos_xy[0], ym - cam_pos_xy[1]
        n = np.hypot(rx, ry)
        rad = (dx * rx + dy * ry) / n
        tan = (-dx * ry + dy * rx) / n
        big = np.hypot(dx, dy) > 30
        small = np.hypot(dx, dy) <= 30
        out[v] = {"rows_dev_gt_30cm": int(np.nansum(big)),
                  "pct_rows_dev_gt_30cm": float(np.nanmean(big) * 100),
                  "pct_pointing_away_from_camera": float(np.nanmean(rad[big] > 0) * 100),
                  "median_abs_radial_cm": float(np.nanmedian(np.abs(rad[big]))),
                  "median_abs_tangential_cm": float(np.nanmedian(np.abs(tan[big]))),
                  "small_noise_std_cm": float(np.nanstd(dx[small]))}
    return out


def smooth_ceiling(d, video="IMG_0104"):
    """TI-HOTA of a perfect-ID tracker whose trajectory is a smoothed copy of the GT."""
    from scipy.ndimage import median_filter
    gt = d[d.video == video][["f", "cx", "cy", "ident"]].sort_values(["ident", "f"]).reset_index(drop=True)
    out = {}
    for w in (5, 9, 15):
        p = gt.copy()
        for ident, idx in p.groupby("ident").groups.items():
            p.loc[idx, "cx"] = median_filter(p.loc[idx, "cx"].values, size=w)
            p.loc[idx, "cy"] = median_filter(p.loc[idx, "cy"].values, size=w)
        out[f"median_{w}"] = ti_hota(gt, p)["TI-HOTA"]
    return out


def cm_per_px_map(h_c2i, ax, title):
    h_i2c = np.linalg.inv(h_c2i)
    xs, ys = np.meshgrid(np.linspace(0, COURT_W, 61), np.linspace(0, COURT_H, 45))
    u, v = apply_h(h_c2i, xs.ravel(), ys.ravel())
    x1, y1 = apply_h(h_i2c, u, v + 1)
    cm = np.hypot(x1 - xs.ravel(), y1 - ys.ravel()).reshape(xs.shape)
    im = ax.contourf(xs, ys, cm, levels=20, cmap="viridis")
    plt.colorbar(im, ax=ax, label="cm per 1 px vertical foot error")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    return cm


def main(data_dir, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    d, stats, h_i2c_train = load(data_dir)
    res = {"per_video": stats}

    # Identity / roster patterns
    roster = pd.read_csv(f"{data_dir}/train_games_roster.csv")
    res["rosters"] = roster[["video", "team: jersey"]].set_index("video")["team: jersey"].to_dict()
    res["final_rosters"] = pd.read_csv(f"{data_dir}/final_games_roster.csv").set_index("video")["team: jersey"].to_dict()

    # Positions
    ing = d[~d.pregame]
    res["position"] = {
        "x_range": [float(d.cx.min()), float(d.cx.max())],
        "y_range": [float(d.cy.min()), float(d.cy.max())],
        "frac_outside_court": float(((d.cx < 0) | (d.cx > COURT_W) | (d.cy < 0) | (d.cy > COURT_H)).mean()),
        "frac_outside_court_ingame": float(((ing.cx < 0) | (ing.cx > COURT_W) | (ing.cy < 0) | (ing.cy > COURT_H)).mean()),
        "frac_outside_by_more_than_1m": float(((d.cx < -100) | (d.cx > COURT_W + 100) | (d.cy < -100) | (d.cy > COURT_H + 100)).mean()),
    }
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (name, sub) in zip(axs, [("all frames", d), ("pre-game", d[d.pregame]), ("in-game", ing)]):
        ax.hist2d(sub.cx, sub.cy, bins=[np.arange(-300, 1810, 25), np.arange(-500, 1410, 25)], cmap="magma", cmin=1)
        court_lines(ax)
        ax.set_title(f"Player density, {name} (n={len(sub)})")
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/position_heatmaps.png", dpi=90)
    plt.close(fig)

    # Kinematics
    kin = kinematics(d)
    ki = kin[~kin.pregame]
    res["speed_mps"] = {"median_ingame": float(ki.speed.median()), "p95_ingame": float(ki.speed.quantile(.95)),
                        "p99_ingame": float(ki.speed.quantile(.99)), "p999_ingame": float(ki.speed.quantile(.999)),
                        "max": float(kin.speed.max()), "median_pregame": float(kin[kin.pregame].speed.median()),
                        "frac_frames_still_lt_0.3mps_ingame": float((ki.speed < 0.3).mean()),
                        "max_cm_per_frame_p999": float(ki.speed.quantile(.999) * 100 / FPS)}
    j_std, j_p95 = gt_jitter(d)
    res["gt_jitter_cm"] = {"residual_std_vs_savgol15": j_std, "residual_abs_p95": j_p95}
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ki.speed.clip(0, 12), bins=120, density=True, alpha=.7, label="in-game")
    ax.hist(kin[kin.pregame].speed.clip(0, 12), bins=120, density=True, alpha=.5, label="pre-game")
    ax.set_xlabel("speed (m/s, frame-to-frame from GT)")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/speed_hist.png", dpi=90)
    plt.close(fig)

    # Proximity
    prox = proximity(d)
    pi = prox[~prox.pregame]
    res["proximity_ingame"] = {
        "frac_frames_any_pair_lt_50cm": float((pi.min_any < 50).mean()),
        "frac_frames_any_pair_lt_100cm": float((pi.min_any < 100).mean()),
        "frac_frames_same_team_lt_100cm": float((pi.min_same < 100).mean()),
        "frac_frames_opp_team_lt_100cm": float((pi.min_opp < 100).mean()),
        "median_min_pair_distance_cm": float(pi.min_any.median()),
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    for col, lab in [("min_same", "closest same-team pair"), ("min_opp", "closest opponent pair")]:
        ax.hist(pi[col].clip(0, 800), bins=80, alpha=.6, label=lab)
    ax.axvline(100, color="k", ls="--", lw=1)
    ax.set_xlabel("distance on court (cm), in-game frames")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/proximity_hist.png", dpi=90)
    plt.close(fig)

    # Bounding boxes in the training view
    o0, o3 = bbox_overlap(d.sample(frac=1.0, random_state=0) if len(d) < 200000 else d[d.f % 5 == 0])
    res["train_view_bbox"] = {"height_px_median": float(d.h.median()), "height_px_p5": float(d.h.quantile(.05)),
                              "height_px_p95": float(d.h.quantile(.95)),
                              "frac_boxes_overlapping_any": float(o0), "frac_boxes_iou_gt_0.3": float(o3)}

    # Camera models
    reg_t = pd.read_csv(f"{data_dir}/train_court_registration.csv")
    reg_f = pd.read_csv(f"{data_dir}/final_court_registration.csv")
    h_t = homography(reg_t[["court_x", "court_y"]].values, reg_t[["image_x", "image_y"]].values)
    h_f = homography(reg_f[["court_x", "court_y"]].values, reg_f[["image_x", "image_y"]].values)
    cam_t = camera_from_homography(h_t, 3840, 2160)
    cam_f = camera_from_homography(h_f, 1280, 720)
    # Validate: predicted pixel height vs annotated box height in the training view.
    smp = d[~d.pregame].sample(20000, random_state=0)
    _, _, hpx, _ = synth_boxes(*cam_t, smp.cx.values, smp.cy.values)
    ratio = smp.h.values / hpx  # implied player height / 172 cm
    res["camera"] = {}
    for name, (k, r, t) in [("train", cam_t), ("final", cam_f)]:
        c = -r.T @ t
        res["camera"][name] = {"focal_px": float(k[0, 0]), "camera_pos_cm": [float(x) for x in c],
                               "height_m": float(abs(c[2]) / 100),
                               "distance_to_court_centre_m": float(np.linalg.norm(c - [COURT_W / 2, COURT_H / 2, 0]) / 100),
                               "depression_angle_deg": float(math.degrees(math.atan2(abs(c[2]), np.linalg.norm(c[:2] - [COURT_W / 2, COURT_H / 2]))))}
    res["camera"]["train_height_check"] = {"implied_player_height_cm_median": float(np.median(ratio) * 172),
                                           "iqr": [float(np.percentile(ratio, 25) * 172), float(np.percentile(ratio, 75) * 172)]}
    ov_f, hp_f = final_view_overlap(d[d.f % 5 == 0], cam_f)
    ov_t, hp_t = final_view_overlap(d[d.f % 5 == 0], cam_t)
    res["simulated_final_view"] = {"player_height_px_median": float(np.median(hp_f)),
                                   "player_height_px_p5": float(np.percentile(hp_f, 5)),
                                   "player_height_px_p95": float(np.percentile(hp_f, 95)),
                                   "frac_boxes_iou_gt_0.3": float(ov_f),
                                   "same_sim_in_train_view_frac_iou_gt_0.3": float(ov_t)}

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    cm_t = cm_per_px_map(h_t, axs[0], "Training view (3840x2160)")
    cm_f = cm_per_px_map(h_f, axs[1], "Final view (1280x720)")
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/cm_per_pixel.png", dpi=90)
    plt.close(fig)
    res["cm_per_px_vertical"] = {"train_median": float(np.median(cm_t)), "train_max": float(cm_t.max()),
                                 "final_median": float(np.median(cm_f)), "final_max": float(cm_f.max())}
    # Same quantity, but weighted by where players actually stand (in-game).
    for name, h in (("train", h_t), ("final", h_f)):
        u, v = apply_h(h, ing.cx.values, ing.cy.values)
        x1, y1 = apply_h(np.linalg.inv(h), u, v + 1)
        cmp = np.hypot(x1 - ing.cx.values, y1 - ing.cy.values)
        res["cm_per_px_vertical"][f"{name}_at_players_median"] = float(np.median(cmp))
        res["cm_per_px_vertical"][f"{name}_at_players_p90"] = float(np.percentile(cmp, 90))

    c_t = -cam_t[1].T @ cam_t[2]
    res["gt_outliers_vs_training_camera_ray"] = gt_outliers(d, c_t[:2])
    res["smoothed_gt_ceiling_IMG_0104"] = smooth_ceiling(d)

    # Where the court and players land in the final image
    fig, ax = plt.subplots(figsize=(12, 7))
    smp2 = d[d.f % 15 == 0]
    fx, fy, hpx, wpx = synth_boxes(*cam_f, smp2.cx.values, smp2.cy.values)
    ax.scatter(fx, fy, s=1, alpha=.15, label="GT foot points (training games re-projected)")
    cx_, cy_ = apply_h(h_f, np.array([0, COURT_W, COURT_W, 0, 0]), np.array([0, 0, COURT_H, COURT_H, 0]))
    ax.plot(cx_, cy_, "k-", label="court boundary")
    ax.set_xlim(0, 1280)
    ax.set_ylim(720, 0)
    ax.set_aspect("equal")
    ax.legend(loc="lower left")
    ax.set_title("Final (challenge) camera: where training-game foot positions would appear")
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/final_view_footprint.png", dpi=90)
    plt.close(fig)

    # Metric sensitivity
    res["sensitivity_IMG_0104"] = sensitivity(d, h_f)
    s = res["sensitivity_IMG_0104"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(list(s["court_noise_cm"]), list(s["court_noise_cm"].values()), "o-")
    axs[0].set_xlabel("Gaussian noise per axis (cm)")
    axs[0].set_ylabel("TI-HOTA (perfect IDs)")
    axs[1].plot(list(s["final_view_pixel_noise"]), list(s["final_view_pixel_noise"].values()), "o-", color="C1")
    axs[1].set_xlabel("foot-point noise in the 1280x720 final view (px per axis)")
    for ax in axs:
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/metric_sensitivity.png", dpi=90)
    plt.close(fig)

    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__)))
