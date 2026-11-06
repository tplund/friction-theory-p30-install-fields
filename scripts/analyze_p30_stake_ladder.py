"""Exploratory read of data/results/p30_stake_ladder_v1.jsonl (design docs/paper30_stake_reading_ladder_design_2026_11_06.md).

Per model x factor x format: the order-averaged avoid logit at each level, and its slope over levels 0-3 with a
bootstrap interval (2,000, seed 0). "Reads" = right sign and interval excluding 0. Own failure rate per level, strict
(the route-B parser: number on the first line) and among answers that contained a number.
"""
import json, sys
from collections import defaultdict
import numpy as np

P = sys.argv[1] if len(sys.argv) > 1 else "data/results/p30_stake_ladder_v1.jsonl"
G = P.replace(".jsonl", ".gen.jsonl")
rows = [json.loads(l) for l in open(P, encoding="utf-8")]
gen = [json.loads(l) for l in open(G, encoding="utf-8")]
models = list(dict.fromkeys(r["model_id"] for r in rows))
RIGHT = {"reward": -1, "tcost": -1, "danger": +1, "difficulty": +1}
av = lambda r: r["option_logprobs"][1] - r["option_logprobs"][0]
rng0 = lambda: np.random.default_rng(0)


def units(mid, factor, fmt):
    """per level: array of order-averaged avoid logits (order 0 and 1 paired by rep index)."""
    d = defaultdict(lambda: {0: [], 1: []})
    for r in rows:
        if r["model_id"] == mid and r["factor"] == factor and r["fmt"] == fmt:
            d[r["level"]][r["order"]].append(av(r))
    out = {}
    for lv, o in d.items():
        out[lv] = np.array(o[0]) if not o[1] else (np.array(o[0]) + np.array(o[1])) / 2
    return out


def slope(u):
    xs = np.arange(4.0)
    pt = np.polyfit(xs, [u[l].mean() for l in range(4)], 1)[0]
    r = rng0()
    bs = [np.polyfit(xs, [u[l][r.integers(0, len(u[l]), len(u[l]))].mean() for l in range(4)], 1)[0] for _ in range(2000)]
    return [round(float(pt), 2), round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]


def order_bias(mid, fmt):
    a = [av(r) for r in rows if r["model_id"] == mid and r["fmt"] == fmt and r["order"] == 0]
    b = [av(r) for r in rows if r["model_id"] == mid and r["fmt"] == fmt and r["order"] == 1]
    return round(float(np.mean(a) - np.mean(b)) / 2, 2) if b else None


out = {}
for mid in models:
    o = {"fail": {}, "factors": {}}
    for lv in (1, 2, 3, 4):
        e = [x for x in gen if x["model_id"] == mid and x["level"] == lv]
        ans = [x for x in e if x["parsed"] is not None]
        o["fail"][lv] = {"strict": round(1 - np.mean([x["correct"] for x in e]), 2),
                         "answered": round(len(ans) / len(e), 2),
                         "given_answer": round(1 - np.mean([x["correct"] for x in ans]), 2) if ans else None}
    for f in RIGHT:
        o["factors"][f] = {}
        for fmt in ("words", "label", "yesno"):
            u = units(mid, f, fmt)
            s = slope(u)
            reads = (s[1] > 0 or s[2] < 0) and np.sign(s[0]) == RIGHT[f]
            wrong = (s[1] > 0 or s[2] < 0) and np.sign(s[0]) != RIGHT[f]
            o["factors"][f][fmt] = {"levels": [round(float(u[l].mean()), 2) for l in range(4)], "slope": s,
                                    "verdict": "reads" if reads else ("WRONG-WAY" if wrong else "flat")}
    o["order_bias"] = {fmt: order_bias(mid, fmt) for fmt in ("words", "label")}
    out[mid] = o

for mid, o in out.items():
    print(f"\n===== {mid}")
    print("  own fail strict / answered / fail|answered:", {lv: (v['strict'], v['answered'], v['given_answer']) for lv, v in o['fail'].items()})
    print("  order bias (half of order0 - order1):", o["order_bias"])
    for f, d in o["factors"].items():
        print(f"  {f:10s} " + " | ".join(f"{fmt}: {v['verdict']:9s} slope {v['slope']} lv {v['levels']}" for fmt, v in d.items()))

print("\nSUMMARY (verdict words / label / yesno)")
for mid, o in out.items():
    print(f"  {mid.split('/')[-1]:20s} " + "  ".join(f"{f}:{'/'.join(o['factors'][f][fmt]['verdict'][:4] for fmt in ('words', 'label', 'yesno'))}" for f in RIGHT))
json.dump(out, open(P.replace(".jsonl", ".summary.json"), "w", encoding="utf-8"), indent=1, default=str)
