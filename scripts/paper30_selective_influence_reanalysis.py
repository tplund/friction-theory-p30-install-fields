"""P30 A.4 — selective-influence re-analysis on existing data (no FT, no API).

Design + falsifiers: docs/paper30_selective_influence_reanalysis_design_premortem_2026_11_04.md
(committed before this script was run).

M: Bradley-Terry additivity test of the Meaning install (N2 v2): is the install's change in
   choice logit a difference of per-band utility changes (drift direction) or a constant tilt
   toward the harder option (starting-point / response bias)?
S: comprehension-time effect at the marker (initiation probe v1.2): locus, sign, instruction.
H: hazard discount v1.1, descriptive.

Usage: python scripts/paper30_selective_influence_reanalysis.py [--data-root C:/_proj/FriktionsLLM]
"""
import argparse, hashlib, json, math, os, random, sys
from collections import defaultdict
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

N2 = "data/install_fields/n2_preference_u_v2_0_run_2026_06_20/preference.jsonl"
N2_DL = "downloads/paper30_n2_preference_u_v2_0_results/preference.jsonl"
INIT = "downloads/initiation_probe_v1_2_results/initiation_v1_2.jsonl"
TRAJ = "data/install_fields/safety_readout_robustness_v1_1_run_2026_06_20/trajectory.jsonl"
HAZ = "downloads/hazard_discount_v1_1_results/choices.jsonl"
BANDS = ["b1", "b2", "b3", "b4", "b5"]
PAIRS = list(combinations(BANDS, 2))
EPS_LEVELS = [1e-2, 1e-3, 1e-4]
EPS_PRIMARY = 1e-3
NBOOT = 2000
SEED = 0


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def logit(q, eps):
    q = min(max(q, eps), 1 - eps)
    return math.log(q / (1 - q))


# ---------- M: Bradley-Terry decomposition ----------
def design_matrix():
    # columns: beta0, u_b2..u_b5 (u_b1 = 0); row per pair (x<y): beta0 + u_y - u_x
    X = []
    for x, y in PAIRS:
        r = [1.0] + [0.0] * 4
        iy, ix = BANDS.index(y), BANDS.index(x)
        if iy > 0: r[iy] += 1.0
        if ix > 0: r[ix] -= 1.0
        X.append(r)
    return np.array(X)


XD = design_matrix()
assert np.linalg.matrix_rank(XD) == 5, "design rank"


def fit_bt(D):
    """D: array (10 pairs, K items). Returns beta0, u (5, u_b1=0), meanD, triangle residual, rss share."""
    y = D.mean(axis=1)                      # pair means (same LS solution as stacking items)
    coef, *_ = np.linalg.lstsq(XD, y, rcond=None)
    beta0 = coef[0]
    u = np.concatenate([[0.0], coef[1:]])
    pm = {p: y[i] for i, p in enumerate(PAIRS)}
    tri = [pm[(a, b)] + pm[(b, c)] - pm[(a, c)] for a, b, c in combinations(BANDS, 3)]
    resid = y - XD @ coef
    return dict(beta0=float(beta0), u=[float(v) for v in u], meanD=float(y.mean()),
                tri=float(np.mean(tri)), rss=float((resid ** 2).sum()), tss=float(((y - y.mean()) ** 2).sum()),
                fit_value_only=None)


def boot(D, nboot=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    K = D.shape[1]
    out = []
    for _ in range(nboot):
        idx = rng.integers(0, K, K)
        out.append(fit_bt(D[:, idx]))
    return out


def ci(vals):
    v = np.sort(np.array(vals))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def summarize(D):
    pt = fit_bt(D)
    bs = boot(D)
    pt["beta0_ci"] = ci([b["beta0"] for b in bs])
    pt["meanD_ci"] = ci([b["meanD"] for b in bs])
    pt["tri_ci"] = ci([b["tri"] for b in bs])
    pt["u_ci"] = [ci([b["u"][j] for b in bs]) for j in range(5)]
    fb = [b["beta0"] / b["meanD"] for b in bs if abs(b["meanD"]) > 1e-9]
    pt["f_bias"] = pt["beta0"] / pt["meanD"] if abs(pt["meanD"]) > 1e-9 else float("nan")
    pt["f_bias_ci"] = ci(fb)
    return pt


def n2_tables(rows, eps):
    # L[model][arm] -> array (10 pairs, 16 items) of logit P(choose harder)
    cell = defaultdict(dict)
    for r in rows:
        if r["phase"] != "main": continue
        q = 1.0 - float(r["p_choose_x"])
        cell[(r["model"], r["arm"])][((r["band_x"], r["band_y"]), int(r["k"]))] = q
    L, clipped = {}, {}
    for (m, a), d in cell.items():
        K = 1 + max(k for (_, k) in d)
        arr = np.full((len(PAIRS), K), np.nan)
        nclip = 0
        for i, p in enumerate(PAIRS):
            for k in range(K):
                q = d.get((p, k))
                if q is None: continue
                if q < eps or q > 1 - eps: nclip += 1
                arr[i, k] = logit(q, eps)
        L.setdefault(m, {})[a] = arr
        clipped.setdefault(m, {})[a] = nclip / arr.size
    return L, clipped


def seedmean(Lm, pole):
    arms = [a for a in Lm if a.startswith(pole)]
    return np.mean([Lm[a] for a in arms], axis=0), arms


def recovery(Lm, observed_meanD, rng):
    """Parameter recovery from base-arm residual noise. (a) pure bias, (b) pure value, same mean."""
    base = Lm["base"]
    resid = base - base.mean(axis=1, keepdims=True)   # item-level noise around pair means
    K = base.shape[1]

    def noise():
        # resample residual columns (items) to keep the cross-pair item structure; difference of two
        # independent draws mimics the variance of a contrast between two arms
        i1, i2 = rng.integers(0, K, K), rng.integers(0, K, K)
        return resid[:, i1] - resid[:, i2]
    gaps = np.array([BANDS.index(y) - BANDS.index(x) for x, y in PAIRS], float)
    out = {}
    for kind in ("pure_bias", "pure_value"):
        fb = []
        for _ in range(200):
            if kind == "pure_bias":
                sig = np.full(len(PAIRS), observed_meanD)
            else:
                sig = gaps * (observed_meanD / gaps.mean())   # u_b linear in band
            D = sig[:, None] + noise()
            f = fit_bt(D)
            fb.append(f["beta0"] / f["meanD"] if abs(f["meanD"]) > 1e-9 else np.nan)
        fb = np.array(fb)
        out[kind] = dict(f_bias_median=float(np.nanmedian(fb)), f_bias_90=[float(np.nanpercentile(fb, 5)), float(np.nanpercentile(fb, 95))])
    return out


def recovery_paired(D, rng):
    """POST-HOC (added after the first pass; see RESULT doc). The pre-specified recovery draws noise from
    two independent base-arm items, which does not cancel item effects; the real contrast is paired on items.
    Here the noise is the observed contrast's own residuals around its pair means, resampled over items."""
    resid = D - D.mean(axis=1, keepdims=True)
    K = D.shape[1]
    meanD = D.mean()
    gaps = np.array([BANDS.index(y) - BANDS.index(x) for x, y in PAIRS], float)
    out = {}
    for kind in ("pure_bias", "pure_value"):
        fb = []
        for _ in range(200):
            sig = np.full(len(PAIRS), meanD) if kind == "pure_bias" else gaps * (meanD / gaps.mean())
            f = fit_bt(sig[:, None] + resid[:, rng.integers(0, K, K)])
            fb.append(f["beta0"] / f["meanD"] if abs(f["meanD"]) > 1e-9 else np.nan)
        fb = np.array(fb)
        out[kind] = dict(f_bias_median=float(np.nanmedian(fb)), f_bias_90=[float(np.nanpercentile(fb, 5)), float(np.nanpercentile(fb, 95))])
    return out


def run_M(root):
    p = os.path.join(root, N2)
    rows = load(p)
    res = {"input": p, "md5": md5(p)}
    pdl = os.path.join(root, N2_DL)
    if os.path.exists(pdl): res["md5_downloads_copy"] = md5(pdl)
    rng = np.random.default_rng(SEED)
    for eps in EPS_LEVELS:
        L, clipped = n2_tables(rows, eps)
        per = {}
        for m in sorted(L):
            Lm = L[m]
            if "base" not in Lm or not any(a.startswith("openness") for a in Lm): continue
            op, op_arms = seedmean(Lm, "openness")
            co, co_arms = seedmean(Lm, "conservatism")
            entry = {"arms_openness": op_arms, "arms_conservatism": co_arms, "clipped_share": clipped[m]}
            if co_arms:
                entry["open_minus_cons"] = summarize(op - co)
            entry["open_minus_base"] = summarize(op - Lm["base"])
            if co_arms:
                entry["cons_minus_base"] = summarize(co - Lm["base"])
            entry["base_bt"] = summarize(Lm["base"])       # base arm itself, descriptive
            # per-seed open-cons where both seeds exist
            for s in ("s1", "s2"):
                a, b = f"openness_{s}", f"conservatism_{s}"
                if a in Lm and b in Lm:
                    entry[f"open_minus_cons_{s}"] = summarize(Lm[a] - Lm[b])
            main = entry.get("open_minus_cons", entry["open_minus_base"])
            entry["gate_pass"] = bool(main["meanD_ci"][0] > 0 or main["meanD_ci"][1] < 0)
            if eps == EPS_PRIMARY and entry["gate_pass"]:
                entry["recovery"] = recovery(Lm, main["meanD"], rng)
                entry["recovery_paired_posthoc"] = recovery_paired(op - co if co_arms else op - Lm["base"], rng)
                rp, rv = entry["recovery"]["pure_bias"]["f_bias_90"], entry["recovery"]["pure_value"]["f_bias_90"]
                entry["recovery_pass_prespecified"] = bool(rp[0] > 0 and rv[1] < 1)
                pp, pv = entry["recovery_paired_posthoc"]["pure_bias"]["f_bias_90"], entry["recovery_paired_posthoc"]["pure_value"]["f_bias_90"]
                entry["recovery_pass_paired"] = bool(pp[0] > 0 and pv[1] < 1)
            # verdict per design note
            b0ci, fb = main["beta0_ci"], main["f_bias"]
            if not entry["gate_pass"]:
                v = "NOT SCORED (gate)"
            elif eps == EPS_PRIMARY and not entry["recovery_pass_prespecified"]:
                v = "NOT SCORED (pre-specified recovery fails)"
            elif (b0ci[0] <= 0 <= b0ci[1]) and abs(fb) < 0.25:
                v = "DRIFT (prediction holds)"
            elif not (b0ci[0] <= 0 <= b0ci[1]) and fb >= 0.5:
                v = "BIAS (falsifier met)"
            else:
                v = "MIXED"
            entry["verdict"] = v
            per[m] = entry
        res[f"eps_{eps:g}"] = per
    return res


# ---------- S: comprehension-time effect ----------
def bci(x, nboot=NBOOT, seed=SEED):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(nboot)]
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def run_S(root):
    p = os.path.join(root, INIT)
    rows = load(p)
    res = {"input": p, "md5": md5(p), "models": {}}
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["arm"], r["cell"])].append(r)
    models = sorted({r["model"] for r in rows})
    for m in models:
        e = {}
        for readout in ("cr_0.10", "marker_entropy"):
            g = lambda arm, cell: [float(r[readout]) for r in sorted(by[(m, arm, cell)], key=lambda r: r["rowid"])]
            d = {}
            for arm in sorted({k[1] for k in by if k[0] == m}):
                d[arm] = {c: float(np.mean(g(arm, c))) for c in ("trained_threat", "trained_neutral", "untrained")}
            inst_t, inst_n = g("installed", "trained_threat"), g("installed", "trained_neutral")
            base_t, base_n = g("base", "trained_threat"), g("base", "trained_neutral")
            cnt_t, cnt_n = g("counter_instructed", "trained_threat"), g("counter_instructed", "trained_neutral")
            # rows are matched by item index within (model, arm, cell): rowid suffix
            def paired(a, b):
                n = min(len(a), len(b)); return np.array(a[:n]) - np.array(b[:n])
            locus = bci(np.concatenate([paired(inst_t, base_t), paired(inst_n, base_n)]))
            # threat and neutral cells hold different probes -> unpaired bootstrap of the mean difference
            rng = np.random.default_rng(SEED); at, an = np.array(inst_t), np.array(inst_n)
            bs = [at[rng.integers(0, len(at), len(at))].mean() - an[rng.integers(0, len(an), len(an))].mean() for _ in range(NBOOT)]
            sign = [float(at.mean() - an.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            instr = bci(np.concatenate([paired(cnt_t, inst_t), paired(cnt_n, inst_n)]))
            ratio = sign[0] / locus[0] if abs(locus[0]) > 1e-9 else float("nan")
            unsigned = (sign[1] <= 0 <= sign[2]) and abs(ratio) < 0.25
            if unsigned: sv = "UNSIGNED (strong A.2 reading falsified)"
            elif sign[2] < 0: sv = "OPPOSITE SIGN: neutral > threat (strong A.2 reading falsified)"
            elif sign[1] > 0: sv = "SIGNED threat > neutral (strong A.2 reading supported)"
            else: sv = "inconclusive"
            # S5 (descriptive, added after first pass): novel untrained nonce in the same slot
            inst_u, base_u = g("installed", "untrained"), g("base", "untrained")
            untr = bci(paired(inst_u, base_u))
            e[readout] = dict(arm_cell_means=d, S1_locus_installed_minus_base=locus,
                              S2_threat_minus_neutral_installed=sign, S2_ratio_to_locus=ratio,
                              S2_verdict=sv, S3_counter_minus_installed=instr,
                              S5_untrained_installed_minus_base=untr, n_per_cell=len(inst_t))
        res["models"][m] = e
    # S4 trajectory
    pt = os.path.join(root, TRAJ)
    if os.path.exists(pt):
        tr = load(pt)
        res["trajectory_input"] = pt; res["trajectory_md5"] = md5(pt)
        agg = defaultdict(lambda: defaultdict(list))
        for r in tr:
            agg[r["model"]][(int(r["offset"]), r["valence"])].append(float(r["cr"]))
        traj = {}
        for m, d in agg.items():
            offs = sorted({o for (o, _) in d})
            traj[m] = {str(o): {"threat": float(np.mean(d[(o, "threat")])), "neutral": float(np.mean(d[(o, "neutral")]))} for o in offs}
        res["S4_trajectory_cr_by_offset"] = traj
    return res


# ---------- H: hazard, descriptive ----------
def run_H(root):
    p = os.path.join(root, HAZ)
    rows = load(p)
    res = {"input": p, "md5": md5(p), "models": {}}
    idx = defaultdict(dict)
    for r in rows:
        key = r["rowid"].split("|")[-1]
        idx[(r["model"], r["condition"])][key] = (float(r["delay"]), logit(float(r["p_sooner"]), EPS_PRIMARY))
    for m in sorted({r["model"] for r in rows}):
        out = {}
        for cond in ("delay_hazard", "flat_hazard"):
            A, B = idx[(m, cond)], idx[(m, "none")]
            ks = sorted(set(A) & set(B))
            x = np.array([math.log(A[k][0]) for k in ks]); d = np.array([A[k][1] - B[k][1] for k in ks])
            X = np.column_stack([np.ones_like(x), x - x.mean()])
            c, *_ = np.linalg.lstsq(X, d, rcond=None)
            rng = np.random.default_rng(SEED); bs = []
            for _ in range(NBOOT):
                i = rng.integers(0, len(ks), len(ks)); cc, *_ = np.linalg.lstsq(X[i], d[i], rcond=None); bs.append(cc)
            bs = np.array(bs)
            out[cond] = dict(n=len(ks), intercept_at_mean_logdelay=[float(c[0])] + ci(bs[:, 0]),
                             slope_per_log_delay=[float(c[1])] + ci(bs[:, 1]))
        res["models"][m] = out
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    a = ap.parse_args()
    have_all = all(os.path.exists(os.path.join(REPO, p)) for p in (N2, INIT, HAZ))
    root = a.data_root or (REPO if have_all else "C:/_proj/FriktionsLLM")   # worktrees lack downloads/
    out = {"script": "scripts/paper30_selective_influence_reanalysis.py",
           "design": "docs/paper30_selective_influence_reanalysis_design_premortem_2026_11_04.md",
           "eps_levels": EPS_LEVELS, "eps_primary": EPS_PRIMARY, "nboot": NBOOT, "seed": SEED,
           "M": run_M(root), "S": run_S(root), "H": run_H(root)}
    od = os.path.join(REPO, "data", "results"); os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, "paper30_selective_influence_reanalysis_v1.json")
    with open(fp, "w", encoding="utf-8") as f: json.dump(out, f, indent=1)
    man = {k: out[k] for k in ("script", "design", "eps_levels", "eps_primary", "nboot", "seed")}
    man["inputs"] = {s: {"path": out[s]["input"], "md5": out[s]["md5"]} for s in ("M", "S", "H")}
    man["inputs"]["S4"] = {"path": out["S"].get("trajectory_input"), "md5": out["S"].get("trajectory_md5")}
    man["substrate_note"] = "local Colab logprob readouts, Qwen2.5 4-bit NF4, deterministic forward passes; no sampling"
    with open(fp.replace(".json", ".manifest.json"), "w", encoding="utf-8") as f: json.dump(man, f, indent=1)
    print("wrote", fp)
    # console summary
    for eps in EPS_LEVELS:
        print(f"\n== M eps={eps:g}")
        for m, e in out["M"][f"eps_{eps:g}"].items():
            mc = e.get("open_minus_cons", e["open_minus_base"])
            print(f"  {m:24s} meanD={mc['meanD']:+.3f} {mc['meanD_ci']}  beta0={mc['beta0']:+.3f} {[round(v,3) for v in mc['beta0_ci']]}  "
                  f"f_bias={mc['f_bias']:+.2f} {[round(v,2) for v in mc['f_bias_ci']]}  tri={mc['tri']:+.3f}  u={[round(v,2) for v in mc['u']]}  -> {e['verdict']}")
            if "recovery" in e:
                print("     recovery (pre-spec):", e["recovery"], "pass", e["recovery_pass_prespecified"])
                print("     recovery (paired, post-hoc):", e["recovery_paired_posthoc"], "pass", e["recovery_pass_paired"])
    print("\n== S")
    for m, e in out["S"]["models"].items():
        for ro, v in e.items():
            print(f"  {m} {ro}: locus={[round(x,3) for x in v['S1_locus_installed_minus_base']]} sign={[round(x,3) for x in v['S2_threat_minus_neutral_installed']]} "
                  f"ratio={v['S2_ratio_to_locus']:+.2f} counter={[round(x,3) for x in v['S3_counter_minus_installed']]} "
                  f"untrained={[round(x,3) for x in v['S5_untrained_installed_minus_base']]} -> {v['S2_verdict']}")
    print("\n== H")
    for m, e in out["H"]["models"].items(): print(" ", m, json.dumps(e))


if __name__ == "__main__":
    main()
