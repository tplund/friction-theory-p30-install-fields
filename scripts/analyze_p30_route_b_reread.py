"""Score data/results/p30_route_b_reread_v1.jsonl per design §8: pick the unsaturated readout by a rule that reads
only base and B-yoked-full, then score H-B (B-own-full minus B-yoked-full, L4 - L1 of the avoid logit) on it."""
import json, sys
from collections import defaultdict
import numpy as np

P = sys.argv[1] if len(sys.argv) > 1 else "data/results/p30_route_b_reread_v1.jsonl"
EP = "data/results/p30_route_b_own_experience_v1.episodes.jsonl"
NB, LIM = 2000, 3.0
rows = [json.loads(l) for l in open(P, encoding="utf-8")]
LV = {}
for l in open(EP, encoding="utf-8"):
    e = json.loads(l); LV[(e["model_id"], e["seed"], e["marker"])] = e["level"]
avoid = lambda r: r["option_logprobs"][1] - r["option_logprobs"][0]
KINDS = ("R1", "R2", "R3", "R4", "stopgo")
models = sorted({r["model_id"] for r in rows})


def mean_logit(mid, arm, kind, seed=None):
    v = [avoid(r) for r in rows if r["model_id"] == mid and r["arm"] == arm and r["kind"] == kind and (seed is None or r["seed"] == seed)]
    return float(np.mean(v))


# selection (base + B-yoked-full only)
sel = {}
for k in KINDS:
    cells = []
    for mid in models:
        cells.append(mean_logit(mid, "base", k))
        for s in (0, 1):
            cells.append(mean_logit(mid, "B-yoked-full", k, s))
    sel[k] = {"cells": [round(c, 2) for c in cells], "max_abs": float(max(abs(c) for c in cells)),
              "unsaturated": all(-LIM <= c <= LIM for c in cells)}
ok = [k for k in KINDS if sel[k]["unsaturated"]]
primary = min(ok, key=lambda k: sel[k]["max_abs"]) if ok else None
print("selection (base + B-yoked-full only):")
for k in KINDS:
    print(f"  {k:7s} max|logit| {sel[k]['max_abs']:6.2f} unsaturated={sel[k]['unsaturated']} cells {sel[k]['cells']}")
print("primary:", primary)


def units(mid, arm_a, arm_b, kind, seeds):
    A = {(r["seed"], r["pid"]): (r["marker"], avoid(r)) for r in rows if r["model_id"] == mid and r["arm"] == arm_a and r["kind"] == kind}
    if arm_b == "base":
        Bb = {r["pid"]: avoid(r) for r in rows if r["model_id"] == mid and r["arm"] == "base" and r["kind"] == kind}
        B = {(s, pid): Bb[pid] for (s, pid) in A}
    else:
        B = {(r["seed"], r["pid"]): avoid(r) for r in rows if r["model_id"] == mid and r["arm"] == arm_b and r["kind"] == kind}
    g = defaultdict(list)
    for (s, pid), (mk, v) in A.items():
        if s in seeds:
            g[LV[(mid, s, mk)]].append(v - B[(s, pid)])
    return {k: np.array(v) for k, v in g.items()}


def boot(g):
    rng = np.random.default_rng(0)
    pt = g[4].mean() - g[1].mean()
    bs = [g[4][rng.integers(0, len(g[4]), len(g[4]))].mean() - g[1][rng.integers(0, len(g[1]), len(g[1]))].mean() for _ in range(NB)]
    return [float(pt), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


f3 = lambda ci: f"{ci[0]:+.3f} [{ci[1]:+.3f},{ci[2]:+.3f}]"
out = {"selection": sel, "primary": primary, "models": {}}
for k in ([primary] if primary else []) + [x for x in KINDS if x != primary]:
    print(f"\n--- readout {k}{' (PRIMARY)' if k == primary else ' (secondary)'}")
    for mid in models:
        o = {}
        o["S_B"] = boot(units(mid, "B-own-full", "B-yoked-full", k, (0, 1)))
        o["S_B_seed"] = {s: boot(units(mid, "B-own-full", "B-yoked-full", k, (s,)))[0] for s in (0, 1)}
        o["H_B_pass"] = o["S_B"][1] > 0 and all(v > 0 for v in o["S_B_seed"].values())
        aseeds = tuple(sorted({r["seed"] for r in rows if r["model_id"] == mid and r["arm"] == "A-desc"}))
        o["S_A"] = boot(units(mid, "A-desc", "base", k, aseeds))
        o["by_level"] = {a: {lv: round(float(np.mean([avoid(r) for r in rows if r["model_id"] == mid and r["arm"] == a and r["kind"] == k
                                                     and LV[(mid, r["seed"] if r["seed"] is not None else 0, r["marker"])] == lv])), 2) for lv in (1, 2, 3, 4)}
                         for a in ("base", "B-own-full", "B-yoked-full", "A-desc")}
        out["models"].setdefault(k, {})[mid] = o
        print(f"  {mid}: S_B {f3(o['S_B'])} seeds {[round(v, 2) for v in o['S_B_seed'].values()]} H-B pass={o['H_B_pass']} | S_A {f3(o['S_A'])}")
        print(f"     by level: {o['by_level']}")
if primary:
    out["verdict"] = {"primary": primary, "H-B per model": {m: out["models"][primary][m]["H_B_pass"] for m in models}}
    print("\nVERDICT:", out["verdict"])
json.dump(out, open(P.replace(".jsonl", ".summary.json"), "w", encoding="utf-8"), indent=1, default=str)
