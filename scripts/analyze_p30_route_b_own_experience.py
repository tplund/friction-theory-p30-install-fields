"""Score data/results/p30_route_b_own_experience_v1.jsonl against the locked criteria in
docs/paper30_route_b_own_experience_design_premortem_2026_11_05.md §4."""
import json, sys
from collections import defaultdict
import numpy as np

P = sys.argv[1] if len(sys.argv) > 1 else "data/results/p30_route_b_own_experience_v1.jsonl"
EP = P.replace(".jsonl", ".episodes.jsonl")
NB = 2000
rows = [json.loads(l) for l in open(P, encoding="utf-8")]
eps = [json.loads(l) for l in open(EP, encoding="utf-8")]
MARKERS = ["kaddow", "nirrel", "oskap", "juvern", "cressit", "dovram", "gorbin", "jarvel"]
NOVEL = ["noddrin", "pamvik", "rusqual", "sevrak"]


def lv_map(mid, seed):
    return {e["marker"]: e["level"] for e in eps if e["model_id"] == mid and e["seed"] == seed}


def logit(r):  # second option vs first
    return r["option_logprobs"][-1] - r["option_logprobs"][0]


def table(R, arm, kind, f=logit):
    """{(seed, pid): (marker, value)}"""
    return {(r["seed"], r["pid"]): (r["marker"], f(r)) for r in R if r["arm"] == arm and r["kind"] == kind}


def contrast_units(R, mid, arm_a, arm_b, kind, f=logit, seeds=(0, 1)):
    """Per (seed,pid) unit: value(a) - value(b) (b may be None), grouped by level in that seed."""
    A = table(R, arm_a, kind, f); B = table(R, arm_b, kind, f) if arm_b else None
    g = defaultdict(list)
    for (s, pid), (mk, v) in A.items():
        if s not in seeds or mk not in MARKERS: continue
        d = v - (B[(s, pid)][1] if B else 0.0)
        g[lv_map(mid, s)[mk]].append(d)
    return {k: np.array(v) for k, v in g.items()}


def boot_level(g):
    rng = np.random.default_rng(0)
    pt = g[4].mean() - g[1].mean()
    bs = [g[4][rng.integers(0, len(g[4]), len(g[4]))].mean() - g[1][rng.integers(0, len(g[1]), len(g[1]))].mean() for _ in range(NB)]
    return [float(pt), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def g2_val(r):  # "almost always too hard" vs "always easy"
    return r["option_logprobs"][3] - r["option_logprobs"][0]


VARIANTS = [("full (primary, addendum §7)", "B-own-full", "B-yoked-full", "tasksize"),
            ("original (G1 failed by design)", "B-own", "B-yoked", "antic")]
f3 = lambda ci: f"{ci[0]:+.3f} [{ci[1]:+.3f},{ci[2]:+.3f}]"
out = {}
for mid in sorted({r["model_id"] for r in rows}):
    R = [r for r in rows if r["model_id"] == mid]
    o = {}
    fail = {}
    for e in eps:
        if e["model_id"] == mid:
            fail.setdefault((e["seed"], e["marker"]), []).append(e["correct"])
    fail = {k: 1 - sum(v) / len(v) for k, v in fail.items()}
    o["own_fail_by_level"] = {lv: float(np.mean([v for (s, mk), v in fail.items() if lv_map(mid, s)[mk] == lv])) for lv in (1, 2, 3, 4)}
    aseeds = sorted({r["seed"] for r in R if r["arm"] == "A-desc"})
    o["G2"] = boot_level(contrast_units(R, mid, "A-desc", None, "g2", g2_val, seeds=aseeds)) if aseeds else None
    G2 = bool(o["G2"] and o["G2"][0] >= 1.0)
    o["S_A"] = boot_level(contrast_units(R, mid, "A-desc", "base", "stopgo", seeds=aseeds)) if aseeds else None
    o["S_P"] = boot_level(contrast_units(R, mid, "P-desc", "base", "stopgo", seeds=aseeds or (0,)))
    print(f"\n=== {mid}")
    print("  own fail by level:", {k: round(v, 3) for k, v in o["own_fail_by_level"].items()})
    if o["G2"]: print(f"  G2 A-desc description: {f3(o['G2'])} pass={G2} | S_A {f3(o['S_A'])} | S_P {f3(o['S_P'])}")
    for vname, own, yok, akind in VARIANTS:
        seeds = sorted({r["seed"] for r in R if r["arm"] == own} & {r["seed"] for r in R if r["arm"] == yok})
        if not seeds:
            continue
        v = {"seeds": seeds}
        v["G1"] = boot_level(contrast_units(R, mid, own, yok, akind, seeds=seeds))
        G1 = v["G1"][0] >= 1.0 and v["G1"][1] > 0
        v["S_B"] = boot_level(contrast_units(R, mid, own, yok, "stopgo", seeds=seeds))
        v["S_B_seed"] = {s: boot_level(contrast_units(R, mid, own, yok, "stopgo", seeds=(s,)))[0] for s in seeds}
        v["H_B_pass"] = G1 and v["S_B"][1] > 0 and all(x > 0 for x in v["S_B_seed"].values())
        pos = lambda ci: bool(ci) and ci[1] > 0
        if not G1: lab = "not scored (G1)"
        else:
            sa = pos(o["S_A"]) if G2 else False; sp = pos(o["S_P"]); sb = pos(v["S_B"])
            lab = ("experience-only" if sb and not (sa or sp) else "experience and description" if sb else
                   "description-only" if (sa or sp) else "neither")
        v["label"] = lab
        A = table(R, own, "stopgo"); B = table(R, yok, "stopgo"); per = defaultdict(list)
        for (s, pid), (mk, val) in A.items():
            if s in seeds and mk in MARKERS: per[(s, mk)].append(val - B[(s, pid)][1])
        keys = sorted(per); x = np.array([fail[k] for k in keys])
        slope = lambda ys: float(np.polyfit(x, ys, 1)[0])
        pt = slope(np.array([np.mean(per[k]) for k in keys])); rng = np.random.default_rng(0); bs = []
        for _ in range(NB):
            bs.append(slope(np.array([np.mean(np.array(per[k])[rng.integers(0, len(per[k]), len(per[k]))]) for k in keys])))
        v["dose_slope"] = [pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        v["dose_points"] = {f"s{k[0]}-{k[1]}": [round(fail[k], 3), round(float(np.mean(per[k])), 3)] for k in keys}
        v["feel_B"] = boot_level(contrast_units(R, mid, own, yok, "feel", seeds=seeds))
        def dst_slope(arm):
            xs, ys = [], []
            for r in R:
                if r["arm"] == arm and r["kind"] == "dst" and r["seed"] in seeds:
                    L = lv_map(mid, r["seed"]); xs.append(L[r["marker2"]] - L[r["marker"]]); ys.append(r["option_logprobs"][0] - r["option_logprobs"][1])
            return float(np.polyfit(xs, ys, 1)[0]) if xs else None  # >0: prefers the path past the easier marker
        v["dst_slope"] = {a: dst_slope(a) for a in (own, yok)}
        nov = lambda arm: float(np.mean([logit(r) for r in R if r["arm"] == arm and r["kind"] == "stopgo" and r["marker"] in NOVEL and r["seed"] in seeds]))
        v["novel_stop_logit"] = {a: nov(a) for a in (own, yok)}
        stop = lambda arm, lv: float(np.mean([logit(r) for r in R if r["arm"] == arm and r["kind"] == "stopgo" and r["seed"] in seeds
                                              and r["marker"] in MARKERS and lv_map(mid, r["seed"])[r["marker"]] == lv]))
        ent = lambda arm, lv: float(np.mean([r["next_entropy_nats_unrenorm_top50"] for r in R if r["arm"] == arm and r["kind"] == "stopgo"
                                             and r["seed"] in seeds and r["marker"] in MARKERS and lv_map(mid, r["seed"])[r["marker"]] == lv]))
        arms = [a for a in ("base", "P-desc", "A-desc", own, yok) if any(r["arm"] == a for r in R)]
        v["stop_logit_by_level"] = {a: {lv: round(stop(a, lv), 3) for lv in (1, 2, 3, 4)} for a in arms}
        v["stopgo_next_entropy"] = {a: {lv: round(ent(a, lv), 3) for lv in (1, 2, 3, 4)} for a in arms}
        o[vname] = v
        print(f"  --- {vname}: {own} vs {yok}, seeds {seeds}")
        print(f"    G1 {akind} L4-L1: {f3(v['G1'])} pass={G1}")
        print(f"    S_B stop-or-go: {f3(v['S_B'])} per seed {[round(x,3) for x in v['S_B_seed'].values()]} H-B pass={v['H_B_pass']} -> label {lab}")
        print(f"    dose slope {f3(v['dose_slope'])} | feel {f3(v['feel_B'])} | dst {v['dst_slope']} | novel {v['novel_stop_logit']}")
        print(f"    stop logit by level {v['stop_logit_by_level']}")
        print(f"    next entropy by level {v['stopgo_next_entropy']}")
    out[mid] = o

P1 = VARIANTS[0][0]
M = [m for m in out if P1 in out[m]]
out["verdict"] = {"variant": P1, "models scored": M,
                  "H-B (primary)": ("supported" if M and all(out[m][P1]["H_B_pass"] for m in M) else "not supported"),
                  "labels": {m: out[m][P1]["label"] for m in M}}
print("\nVERDICT:", out["verdict"])
json.dump(out, open(P.replace(".jsonl", ".summary.json"), "w", encoding="utf-8"), indent=1, default=str)
