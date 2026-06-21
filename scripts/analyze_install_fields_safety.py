#!/usr/bin/env python
"""
analyze_install_fields_safety.py
================================
Local analyzer for the Safety-install notebook (notebook 1) output. Reads the unzipped Drive
run dir (install_fields_safety_v*_data/) and reports the three load-bearing results with paired
significance, plus the REGEL-8 version-set guard.

Design: docs/install_fields_safety_install_design_2026_06_17.md

Readout is the WITHIN-PAIR contrast Δ = stat(threat) - stat(neutral), paired by pair_id (same
activity frame; only the object differs). We test it paired across pairs.

Results reported:
  1. INSTALL          PRE-base vs POST-FT Δfirst_cr on threat_A_held (does the function install?)
  2. FUNCTION-TRANSFER POST-FT Δ on threat_B_held, held-out categories (function, not facts?)
  3. MODULABILITY     POST-FT normal vs ignore-instruction Δ (capture suppressible = not fast involuntary?)

Usage:
  python scripts/analyze_install_fields_safety.py --selftest
  python scripts/analyze_install_fields_safety.py --dir downloads/install_fields_safety_v1_0_data
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

try:  # Windows cp1252 consoles choke on Δ; force utf-8 output
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

STAT = "first_cr"  # CR readout — SECONDARY (route-count is blind to orient-vs-proceed synonym clusters)

# Content-aware cluster readout — PRIMARY (v1-run 2026-06-17 found first_cr masked the install:
# competing orient-synonyms and competing proceed-synonyms both raise route-count equally).
ORIENT_TOKENS = {"Stop", "Hold", "Wait", "Pause", "Assess", "Don", "Stay", "Be", "Care", "Careful"}
PROCEED_TOKENS = {"Nothing", "Proceed", "Continue", "There", "Everything", "This", "Fine", "No", "Keep", "It", "I"}


def _cluster_masses(row):
    """From raw_first_top, return (orient_mass, proceed_mass) at the first decision token."""
    o = p = 0.0
    for t, lp in row.get("raw_first_top", []):
        s = t.strip()
        if s in ORIENT_TOKENS:
            o += math.exp(lp)
        elif s in PROCEED_TOKENS:
            p += math.exp(lp)
    return o, p


def _cluster_entropy(o, p):
    """2-cluster entropy = how OPEN the orient-vs-proceed race is (P13 race-opening)."""
    tot = o + p
    if tot <= 0:
        return 0.0
    a, b = o / tot, p / tot
    return -sum(x * math.log(x) for x in (a, b) if x > 0)


def cluster_orient_gaps(rows, condition):
    """Within-pair (threat - neutral) orient-mass gaps for a condition (needs raw_first_top)."""
    byp = defaultdict(dict)
    for r in rows:
        byp[r["pair_id"]][r["valence"]] = r
    gaps, race_th, race_ne = [], [], []
    for pid, m in byp.items():
        if "threat" in m and "neutral" in m and m["threat"].get("condition") == condition \
                and m["threat"].get("raw_first_top") is not None:
            ot, pt = _cluster_masses(m["threat"])
            on, pn = _cluster_masses(m["neutral"])
            gaps.append(ot - on)
            race_th.append(_cluster_entropy(ot, pt))
            race_ne.append(_cluster_entropy(on, pn))
    return gaps, race_th, race_ne


def load(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def within_pair_deltas(rows, condition, stat=STAT):
    """Return list of (threat-neutral) deltas paired by pair_id for one condition."""
    byp = defaultdict(dict)
    for r in rows:
        byp[r["pair_id"]][r["valence"]] = r
    out = []
    for pid, m in byp.items():
        if "threat" in m and "neutral" in m and m["threat"].get("condition") == condition:
            out.append(m["threat"][stat] - m["neutral"][stat])
    return out


def paired_against(deltas_a, deltas_b):
    """Mean difference (a-b) — here used pairwise where lengths match; falls back to
    independent mean-diff if pairing is unavailable."""
    if deltas_a and deltas_b and len(deltas_a) == len(deltas_b):
        diffs = [a - b for a, b in zip(deltas_a, deltas_b)]
        return _summ(diffs)
    return {"mean": _mean(deltas_a) - _mean(deltas_b), "n": min(len(deltas_a), len(deltas_b)),
            "t": float("nan"), "p_approx": float("nan"), "ci95": (float("nan"), float("nan"))}


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _summ(xs):
    """One-sample paired summary: mean, t-stat vs 0, ~p (normal approx), 95% CI."""
    n = len(xs)
    if n == 0:
        return {"mean": float("nan"), "n": 0, "t": float("nan"), "p_approx": float("nan"),
                "ci95": (float("nan"), float("nan"))}
    m = _mean(xs)
    if n == 1:
        return {"mean": m, "n": 1, "t": float("nan"), "p_approx": float("nan"), "ci95": (m, m)}
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    se = sd / math.sqrt(n) if sd > 0 else 0.0
    t = m / se if se > 0 else float("inf")
    # two-sided normal approximation to the p-value (good enough for a quick read; the
    # notebook is the instrument, this is triage)
    p = math.erfc(abs(t) / math.sqrt(2)) if se > 0 else 0.0
    ci = (m - 1.96 * se, m + 1.96 * se)
    return {"mean": m, "n": n, "t": t, "p_approx": p, "ci95": ci}


def version_guard(*rowsets):
    vs = set()
    for rows in rowsets:
        for r in rows:
            vs.add((r.get("nb_version"), r.get("tag")))
    return vs


def report(d):
    pre = load(os.path.join(d, "readout_pre_base.jsonl"))
    post = load(os.path.join(d, "readout_post_ft.jsonl"))
    ign = load(os.path.join(d, "readout_post_ft_ignore.jsonl"))
    traj = load(os.path.join(d, "training_trajectory.jsonl"))

    vs = version_guard(pre, post, ign, traj)
    print(f"REGEL-8 version-set: {vs}", "  ** MIXED VERSIONS **" if len(vs) > 1 else "  (clean)")
    if not (pre and post):
        print("MISSING pre/post readout files — nothing to analyze.")
        return

    # condition-agnostic: any non-neutral threat condition present in the data (handles both the
    # real arm threat_A_held/threat_B_held and the invented arm invented_held/invented_seen)
    conditions = sorted({r["condition"] for r in post if r.get("valence") == "threat"})
    print(f"\nthreat conditions detected: {conditions}")

    # ---- PRIMARY readout: content-aware orient-cluster mass + race-openness ----
    if any(r.get("raw_first_top") is not None for r in post):
        print("\n=== PRIMARY: orient-cluster mass at first token (threat - neutral, within-pair) ===")
        for c in conditions:
            g_post, rth, rne = cluster_orient_gaps(post, c)
            g_pre, _, _ = cluster_orient_gaps(pre, c)
            print(f"  {c:16s}: PRE {_mean(g_pre):+.3f} | POST {fmt(_summ(g_post))}")
            if rth:
                print(f"  {'':16s}  race-openness (cluster-entropy): threat {_mean(rth):.3f} vs neutral {_mean(rne):.3f}")
        print("  -> install/transfer if POST gap > 0 (p<.05); higher threat cluster-entropy = race held open")

    print("\n=== SECONDARY (first_cr — route-count, masked by synonyms): INSTALL + TRANSFER ===")
    for c in conditions:
        c_pre = within_pair_deltas(pre, c)
        c_post = within_pair_deltas(post, c)
        lift = paired_against(c_post, c_pre)
        print(f"  {c:16s}: PRE {_mean(c_pre):+.3f} | POST {fmt(_summ(c_post))}")
        print(f"  {'':16s}  LIFT {fmt(lift)}  -> installs/transfers if mean>0 & p<.05")
    print("  (real arm: install=threat_A_held, function-not-facts=threat_B_held transfer;")
    print("   invented arm: function=invented_held generalises ~ invented_seen, confound-free)")

    print("\n=== 3. MODULABILITY (POST normal vs ignore-instruction) ===")
    ign_has_raw = ign and any(r.get("raw_first_top") is not None for r in ign)
    for c in conditions:
        if ign_has_raw:  # PRIMARY: cluster orient-mass gap (paired)
            norm, _, _ = cluster_orient_gaps(post, c)
            igd, _, _ = cluster_orient_gaps(ign, c)
            metric = "orient-mass"
        else:            # fall back to first_cr if the ignore run didn't save raw_first_top
            norm = within_pair_deltas(post, c)
            igd = within_pair_deltas(ign, c)
            metric = "first_cr"
        supp = paired_against(norm, igd)
        print(f"  {c:14s} [{metric}]: normal {_mean(norm):+.3f} -> ignore {_mean(igd):+.3f} | suppression {fmt(supp)}")
    print("  -> capture MODULABLE (suppression>0 & p<.05) = NOT fast involuntary (split-prediction)")

    if traj:
        steps = sorted({r["step"] for r in traj})
        print(f"\n=== trajectory: {len(steps)} checkpoints, steps {steps[0]}..{steps[-1]} ===")
        tkey = "orient_mass" if any("orient_mass" in r for r in traj) else "first_cr"
        for s in steps:
            srows = [r for r in traj if r["step"] == s]
            byp = defaultdict(dict)
            for r in srows:
                byp[r["pair_id"]][r["valence"]] = r[tkey]
            ds = [m["threat"] - m["neutral"] for m in byp.values() if "threat" in m and "neutral" in m]
            print(f"  step {s:4d}: Δ{tkey}(threat-neutral) = {_mean(ds):+.3f}")


def fmt(s):
    return (f"mean {s['mean']:+.3f}  95%CI[{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}]  "
            f"t={s['t']:.2f} p~{s['p_approx']:.3g} (n={s['n']})")


def selftest():
    """Synthetic rows: install raises threat first_cr; transfers to B; ignore suppresses it."""
    import tempfile
    d = tempfile.mkdtemp()

    def write(name, phase, threat_cr_A, threat_cr_B, neutral_cr):
        with open(os.path.join(d, name), "w") as f:
            pid = 0
            for cond, tcr in [("threat_A_held", threat_cr_A), ("threat_B_held", threat_cr_B)]:
                for _ in range(12):
                    f.write(json.dumps({"nb_version": "1.0", "tag": "t", "phase": phase, "pair_id": pid,
                            "condition": cond, "category": "x", "valence": "threat", "first_cr": tcr,
                            "first_entropy": 1.0}) + "\n")
                    f.write(json.dumps({"nb_version": "1.0", "tag": "t", "phase": phase, "pair_id": pid,
                            "condition": cond, "category": "x", "valence": "neutral", "first_cr": neutral_cr,
                            "first_entropy": 1.0}) + "\n")
                    pid += 1

    write("readout_pre_base.jsonl", "pre_base", 1.6, 1.55, 1.5)     # tiny pre gap
    write("readout_post_ft.jsonl", "post_ft", 2.8, 2.6, 1.5)        # install + transfer
    write("readout_post_ft_ignore.jsonl", "ignore", 1.7, 1.65, 1.5) # ignore suppresses
    report(d)
    # assertions
    a_post = within_pair_deltas(load(os.path.join(d, "readout_post_ft.jsonl")), "threat_A_held")
    assert _mean(a_post) > 1.0, "synthetic install should show large Δ"
    b_post = within_pair_deltas(load(os.path.join(d, "readout_post_ft.jsonl")), "threat_B_held")
    assert _mean(b_post) > 1.0, "synthetic transfer should show large Δ"
    print("\nSELFTEST PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.dir:
        report(args.dir)
    else:
        ap.error("pass --dir <run dir> or --selftest")


if __name__ == "__main__":
    main()
