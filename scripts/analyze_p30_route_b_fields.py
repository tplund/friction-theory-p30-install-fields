"""Exploratory read of data/results/p30_route_b_fields_v1.jsonl (design §9).

For each model: the avoid level per arm and format; the hard-minus-easy contrast (L4 - L1 of the avoid logit,
order-averaged), own minus yoked, per stake cell and format, with its slope over reward and time cost; the
expected-outcome readouts. Bootstrap over probes (2,000, seed 0). Nothing here is a locked criterion.
"""
import json, sys
from collections import defaultdict
import numpy as np

P = sys.argv[1] if len(sys.argv) > 1 else "data/results/p30_route_b_fields_v1.jsonl"
EP = "data/results/p30_route_b_own_experience_v1.episodes.jsonl"
NB = 2000
rows = [json.loads(l) for l in open(P, encoding="utf-8")]
LV, FAIL = {}, defaultdict(list)
for l in open(EP, encoding="utf-8"):
    e = json.loads(l)
    LV[(e["model_id"], e["seed"], e["marker"])] = e["level"]
    FAIL[(e["model_id"], e["seed"], e["marker"])].append(0 if e["correct"] else 1)
FR = {k: float(np.mean(v)) for k, v in FAIL.items()}
av = lambda r: r["option_logprobs"][1] - r["option_logprobs"][0]
models = sorted({r["model_id"] for r in rows})
RW, TC = (0, 2, 10), (0, 10, 60)
out = {}


def lv(mid, r):
    return LV[(mid, r["seed"] if r["seed"] is not None else 0, r["marker"])]


def decide_units(mid, arm, fmt, seed):
    """order-averaged avoid logit per probe unit: key (marker, reward, tcost, rep) -> value.
    A probe unit pairs order 0 and order 1 by their position within the cell (same rep index)."""
    cell = defaultdict(lambda: {0: [], 1: []})
    for r in rows:
        if r["model_id"] == mid and r["arm"] == arm and r["kind"] == "decide" and r["fmt"] == fmt and (arm == "base" or r["seed"] == seed):
            cell[(r["marker"], r["reward"], r["tcost"])][r["order"]].append(av(r))
    u = {}
    for k, d in cell.items():
        for i, (a, b) in enumerate(zip(d[0], d[1])):
            u[k + (i,)] = (a + b) / 2
    return u


def contrast(mid, arm_a, arm_b, fmt, seeds, cells=None):
    """L4 - L1 of (arm_a - arm_b), per probe unit; returns {level: array} pooled over seeds."""
    g = defaultdict(list)
    for s in seeds:
        A = decide_units(mid, arm_a, fmt, s); B = decide_units(mid, arm_b, fmt, s)
        for k, v in A.items():
            if cells and (k[1], k[2]) not in cells:
                continue
            g[LV[(mid, s, k[0])]].append(v - B[k])
    return {k: np.array(v) for k, v in g.items()}


def boot_diff(g, rng):
    pt = g[4].mean() - g[1].mean()
    bs = [g[4][rng.integers(0, len(g[4]), len(g[4]))].mean() - g[1][rng.integers(0, len(g[1]), len(g[1]))].mean() for _ in range(NB)]
    return [round(float(pt), 2), round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]


def slope(mid, arm_a, arm_b, fmt, seeds, axis):
    """slope of the L4-L1 own-minus-yoked contrast over reward (coins) or time cost (minutes), via bootstrap."""
    xs = RW if axis == "reward" else TC
    per = {}
    for x in xs:
        cells = [(x, t) for t in TC] if axis == "reward" else [(r, x) for r in RW]
        per[x] = contrast(mid, arm_a, arm_b, fmt, seeds, cells)
    rng = np.random.default_rng(0)
    def fit(get):
        y = [get(per[x]) for x in xs]
        return np.polyfit(np.array(xs, float), np.array(y), 1)[0]
    pt = fit(lambda g: g[4].mean() - g[1].mean())
    bs = [fit(lambda g: g[4][rng.integers(0, len(g[4]), len(g[4]))].mean() - g[1][rng.integers(0, len(g[1]), len(g[1]))].mean()) for _ in range(NB)]
    unit = 10 if axis == "reward" else 60
    return {"per_x": {x: round(float(per[x][4].mean() - per[x][1].mean()), 2) for x in xs},
            f"slope_per_{unit}": [round(float(pt * unit), 2), round(float(np.percentile(bs, 2.5) * unit), 2), round(float(np.percentile(bs, 97.5) * unit), 2)]}


def expect(mid, arm_a, arm_b, kind, seeds):
    g = defaultdict(list)
    for s in seeds:
        A = {r["pid"]: (r["marker"], av(r)) for r in rows if r["model_id"] == mid and r["arm"] == arm_a and r["kind"] == kind and r["seed"] == s}
        if arm_b == "base":
            B = {r["pid"]: av(r) for r in rows if r["model_id"] == mid and r["arm"] == "base" and r["kind"] == kind}
        else:
            B = {r["pid"]: av(r) for r in rows if r["model_id"] == mid and r["arm"] == arm_b and r["kind"] == kind and r["seed"] == s}
        for pid, (mk, v) in A.items():
            g[LV[(mid, s, mk)]].append(v - B[pid])
    return boot_diff({k: np.array(v) for k, v in g.items()}, np.random.default_rng(0))


def marker_table(mid, fmt, seed):
    """per marker: own failure rate, avoid own, avoid yoked (order-averaged, all cells), outcome own-yoked."""
    o = decide_units(mid, "B-own-full", fmt, seed); y = decide_units(mid, "B-yoked-full", fmt, seed)
    t = []
    for mk in sorted({k[0] for k in o}, key=lambda m: (LV[(mid, seed, m)], m)):
        ko = [v for k, v in o.items() if k[0] == mk]; ky = [v for k, v in y.items() if k[0] == mk]
        t.append({"marker": mk, "level": LV[(mid, seed, mk)], "fail": round(FR[(mid, seed, mk)], 2),
                  "own": round(float(np.mean(ko)), 2), "yoked": round(float(np.mean(ky)), 2)})
    return t


for mid in models:
    o = {}
    aseeds = tuple(sorted({r["seed"] for r in rows if r["model_id"] == mid and r["arm"] == "A-desc"}))
    # absolute levels
    lev = {}
    for arm in ("base", "B-own-full", "B-yoked-full", "A-desc"):
        for fmt in ("letter", "words"):
            v = [av(r) for r in rows if r["model_id"] == mid and r["arm"] == arm and r["kind"] == "decide" and r["fmt"] == fmt]
            v0 = [av(r) for r in rows if r["model_id"] == mid and r["arm"] == arm and r["kind"] == "decide" and r["fmt"] == fmt and r["order"] == 0]
            v1 = [av(r) for r in rows if r["model_id"] == mid and r["arm"] == arm and r["kind"] == "decide" and r["fmt"] == fmt and r["order"] == 1]
            lev[f"{arm}|{fmt}"] = {"mean": round(float(np.mean(v)), 2), "order0": round(float(np.mean(v0)), 2), "order1": round(float(np.mean(v1)), 2)}
        for kind in ("outcome", "ease"):
            v = [av(r) for r in rows if r["model_id"] == mid and r["arm"] == arm and r["kind"] == kind]
            lev[f"{arm}|{kind}"] = round(float(np.mean(v)), 2)
    o["levels"] = lev
    # stake-dependence of the base's own avoid logit (sanity: does the base model read the stakes at all?)
    sb = {}
    for fmt in ("letter", "words"):
        u = decide_units(mid, "base", fmt, None)
        sb[fmt] = {f"rw{rw}": round(float(np.mean([v for k, v in u.items() if k[1] == rw])), 2) for rw in RW}
        sb[fmt].update({f"tc{tc}": round(float(np.mean([v for k, v in u.items() if k[2] == tc])), 2) for tc in TC})
    o["base_stake_reading"] = sb
    for fmt in ("letter", "words"):
        f = {}
        f["S_B"] = boot_diff(contrast(mid, "B-own-full", "B-yoked-full", fmt, (0, 1)), np.random.default_rng(0))
        f["S_B_seed"] = {s: boot_diff(contrast(mid, "B-own-full", "B-yoked-full", fmt, (s,)), np.random.default_rng(0))[0] for s in (0, 1)}
        f["S_A"] = boot_diff(contrast(mid, "A-desc", "base", fmt, aseeds), np.random.default_rng(0))
        f["cells"] = {f"rw{rw}_tc{tc}": round(float((lambda g: g[4].mean() - g[1].mean())(contrast(mid, "B-own-full", "B-yoked-full", fmt, (0, 1), [(rw, tc)]))), 2)
                      for rw in RW for tc in TC}
        f["reward"] = slope(mid, "B-own-full", "B-yoked-full", fmt, (0, 1), "reward")
        f["tcost"] = slope(mid, "B-own-full", "B-yoked-full", fmt, (0, 1), "tcost")
        f["markers_seed0"] = marker_table(mid, fmt, 0)
        f["markers_seed1"] = marker_table(mid, fmt, 1)
        o[fmt] = f
    for kind in ("outcome", "ease"):
        o[kind] = {"S_B": expect(mid, "B-own-full", "B-yoked-full", kind, (0, 1)),
                   "S_B_seed": {s: expect(mid, "B-own-full", "B-yoked-full", kind, (s,))[0] for s in (0, 1)},
                   "S_A": expect(mid, "A-desc", "base", kind, aseeds)}
    out[mid] = o

for mid, o in out.items():
    print(f"\n===== {mid}")
    print("levels:", json.dumps(o["levels"]))
    print("base reads stakes:", json.dumps(o["base_stake_reading"]))
    for fmt in ("letter", "words"):
        f = o[fmt]
        print(f"[{fmt}] S_B {f['S_B']} seeds {f['S_B_seed']} | S_A {f['S_A']}")
        print(f"   reward {f['reward']}  tcost {f['tcost']}")
        print(f"   cells {f['cells']}")
        print(f"   markers s0 {[(m['level'], m['fail'], m['own'], m['yoked']) for m in f['markers_seed0']]}")
        print(f"   markers s1 {[(m['level'], m['fail'], m['own'], m['yoked']) for m in f['markers_seed1']]}")
    for kind in ("outcome", "ease"):
        print(f"[{kind}] S_B {o[kind]['S_B']} seeds {o[kind]['S_B_seed']} | S_A {o[kind]['S_A']}")
json.dump(out, open(P.replace(".jsonl", ".summary.json"), "w", encoding="utf-8"), indent=1, default=str)
