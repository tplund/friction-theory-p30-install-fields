"""Exploratory read of data/results/p30_route_b_instruct_v1.jsonl (design §10 of
docs/paper30_route_b_own_experience_design_premortem_2026_11_05.md).

Primary: can an experienced good match override Safety (danger) and Meaning (reward)?
  d = avoid logit (own) - avoid logit (yoked), paired by probe, order-averaged.
  I_danger = slope_danger(d | L4 markers) - slope_danger(d | L1 markers); positive = good match damps danger.
  I_reward = slope_reward(d | L4) - slope_reward(d | L1); negative = reward matters more where the match is poor.
Also: G1 (tasksize), outcome, g2, S_B per cell and format, base positive control, A-desc. Bootstrap 2,000, seed 0.
"""
import json, sys
from collections import defaultdict
import numpy as np

P = sys.argv[1] if len(sys.argv) > 1 else "data/results/p30_route_b_instruct_v1.jsonl"
EP = P.replace(".jsonl", ".episodes.jsonl")
rows = [json.loads(l) for l in open(P, encoding="utf-8")]
LV, FAIL = {}, defaultdict(list)
for l in open(EP, encoding="utf-8"):
    e = json.loads(l); LV[(e["seed"], e["marker"])] = e["level"]; FAIL[(e["seed"], e["marker"])].append(1 - e["correct"])
av = lambda r: r["option_logprobs"][1] - r["option_logprobs"][0]
NB = 2000
FMTS = ("option", "yesno", "words")


def dunits(arm, fmt, seed):
    """order-averaged avoid logit per unit key (marker, danger, reward, rep)."""
    c = defaultdict(lambda: {0: [], 1: []})
    for r in rows:
        if r["kind"] == "decide" and r["arm"] == arm and r["fmt"] == fmt and (arm == "base" or r["seed"] == seed):
            c[(r["marker"], r["danger"], r["reward"])][r["order"]].append(av(r))
    u = {}
    for k, d in c.items():
        for i in range(len(d[0])):
            u[k + (i,)] = (d[0][i] + d[1][i]) / 2 if d[1] else d[0][i]
    return u


def paired(arm_a, arm_b, fmt, seeds):
    """list of (seed, level, danger, reward, value a - b)."""
    out = []
    for s in seeds:
        A = dunits(arm_a, fmt, s); B = dunits(arm_b, fmt, s)
        for k, v in A.items():
            out.append((s, LV[(s, k[0])], k[1], k[2], v - B[k]))
    return out


def stats(units, rng=None):
    """from (seed, level, danger, reward, d) units: S (L4-L1), slopes by level, interactions."""
    g = defaultdict(list)
    for s, lv, dg, rw, d in units:
        g[(lv, dg, rw)].append(d)
    g = {k: np.array(v) for k, v in g.items()}
    if rng is not None:
        g = {k: v[rng.integers(0, len(v), len(v))] for k, v in g.items()}
    m = lambda lv, dg=None, rw=None: np.mean(np.concatenate([v for (l, d_, r_), v in g.items() if l == lv and (dg is None or d_ == dg) and (rw is None or r_ == rw)]))
    xs = np.arange(4.0)
    sl = lambda lv, axis: np.polyfit(xs, [m(lv, dg=x) if axis == "danger" else m(lv, rw=x) for x in range(4)], 1)[0]
    return {"S": m(4) - m(1), "slope_danger_L1": sl(1, "danger"), "slope_danger_L4": sl(4, "danger"),
            "slope_reward_L1": sl(1, "reward"), "slope_reward_L4": sl(4, "reward"),
            "I_danger": sl(4, "danger") - sl(1, "danger"), "I_reward": sl(4, "reward") - sl(1, "reward")}


def boot(units):
    pt = stats(units); rng = np.random.default_rng(0)
    bs = [stats(units, rng) for _ in range(NB)]
    return {k: [round(float(pt[k]), 2), round(float(np.percentile([b[k] for b in bs], 2.5)), 2), round(float(np.percentile([b[k] for b in bs], 97.5)), 2)] for k in pt}


def arm_slopes(arm, fmt, seeds):
    """absolute slopes (no subtraction) of one arm: danger and reward, pooled over markers, plus by level."""
    us = []
    for s in seeds:
        for k, v in dunits(arm, fmt, s).items():
            us.append((s, LV[(s if s is not None else 0, k[0])], k[1], k[2], v))
    g = defaultdict(list)
    for s, lv, dg, rw, v in us:
        g[(dg, "d")].append(v); g[(rw, "r")].append(v)
    xs = np.arange(4.0)
    return {"mean": round(float(np.mean([u[4] for u in us])), 2),
            "slope_danger": round(float(np.polyfit(xs, [np.mean(g[(x, "d")]) for x in range(4)], 1)[0]), 2),
            "slope_reward": round(float(np.polyfit(xs, [np.mean(g[(x, "r")]) for x in range(4)], 1)[0]), 2),
            "by_level": {lv: round(float(np.mean([u[4] for u in us if u[1] == lv])), 2) for lv in (1, 2, 3, 4)}}


def simple(kind, arm_a, arm_b, seeds, opt_idx=(1, 0)):
    g = defaultdict(list)
    for s in seeds:
        A = {r["pid"]: r for r in rows if r["kind"] == kind and r["arm"] == arm_a and r["seed"] == s}
        B = {r["pid"]: r for r in rows if r["kind"] == kind and r["arm"] == arm_b and (arm_b == "base" or r["seed"] == s)}
        for pid, r in A.items():
            f = lambda x: x["option_logprobs"][opt_idx[0]] - x["option_logprobs"][opt_idx[1]]
            g[LV[(s, r["marker"])]].append(f(r) - f(B[pid]))
    g = {k: np.array(v) for k, v in g.items()}
    rng = np.random.default_rng(0)
    pt = g[4].mean() - g[1].mean()
    bs = [g[4][rng.integers(0, len(g[4]), len(g[4]))].mean() - g[1][rng.integers(0, len(g[1]), len(g[1]))].mean() for _ in range(NB)]
    return [round(float(pt), 2), round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]


out = {"fail_by_level": {s: {lv: round(float(np.mean([v for (ss, mk), vv in FAIL.items() if ss == s and LV[(ss, mk)] == lv for v in vv])), 3) for lv in (1, 2, 3, 4)} for s in (0, 1)}}
out["G1_tasksize"] = {"pooled": simple("tasksize", "B-own-full", "B-yoked-full", (0, 1)),
                      "seeds": {s: simple("tasksize", "B-own-full", "B-yoked-full", (s,))[0] for s in (0, 1)}}
out["outcome_fail_minus_succeed"] = {"S_B": simple("outcome", "B-own-full", "B-yoked-full", (0, 1)),
                                     "seeds": {s: simple("outcome", "B-own-full", "B-yoked-full", (s,))[0] for s in (0, 1)},
                                     "S_A": simple("outcome", "A-desc", "base", (0,))}
out["g2_desc_recite"] = {"A-desc_minus_base": simple("g2", "A-desc", "base", (0,), opt_idx=(3, 0))}
for fmt in FMTS:
    f = {"base": arm_slopes("base", fmt, (None,)),
         "own": {s: arm_slopes("B-own-full", fmt, (s,)) for s in (0, 1)},
         "yoked": {s: arm_slopes("B-yoked-full", fmt, (s,)) for s in (0, 1)},
         "A-desc": arm_slopes("A-desc", fmt, (0,)),
         "own_minus_yoked": boot(paired("B-own-full", "B-yoked-full", fmt, (0, 1))),
         "own_minus_yoked_seed": {s: {k: v[0] for k, v in boot(paired("B-own-full", "B-yoked-full", fmt, (s,))).items()} for s in (0, 1)},
         "A-desc_minus_base": boot(paired("A-desc", "base", fmt, (0,)))}
    # S_B per (danger, reward) cell
    U = paired("B-own-full", "B-yoked-full", fmt, (0, 1))
    f["S_B_cells"] = {f"dg{dg}_rw{rw}": round(float(np.mean([u[4] for u in U if u[1] == 4 and u[2] == dg and u[3] == rw])
                                                   - np.mean([u[4] for u in U if u[1] == 1 and u[2] == dg and u[3] == rw])), 2)
                      for dg in range(4) for rw in range(4)}
    out[fmt] = f

print(json.dumps({k: out[k] for k in ("fail_by_level", "G1_tasksize", "outcome_fail_minus_succeed", "g2_desc_recite")}, indent=None))
for fmt in FMTS:
    f = out[fmt]
    print(f"\n=== {fmt}")
    print("  base:", f["base"])
    print("  own s0/s1:", f["own"][0], f["own"][1])
    print("  yoked s0/s1:", f["yoked"][0], f["yoked"][1])
    print("  A-desc:", f["A-desc"])
    print("  OWN - YOKED:", f["own_minus_yoked"])
    print("  per seed:", f["own_minus_yoked_seed"])
    print("  A-desc - base:", f["A-desc_minus_base"])
    print("  S_B cells:", f["S_B_cells"])
json.dump(out, open(P.replace(".jsonl", ".summary.json"), "w", encoding="utf-8"), indent=1, default=str)
