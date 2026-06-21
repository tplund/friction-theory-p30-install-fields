#!/usr/bin/env python
"""Analyze Install-fields MEANING v1 readouts (A preference+conflict / B boredom / C willingness-to-pay).

Order-counterbalanced: readA highA/highB, readC yesA/yesB -> average over order per (model,condition,pair)
to remove POSITION bias before reading the value signal (base shows pure position bias: p_high 0.99 vs 0.006).

Usage: python scripts/analyze_install_fields_meaning.py <results_dir>
"""
import json, sys, collections, math, os

D = sys.argv[1] if len(sys.argv) > 1 else "."

def load(fn):
    return [json.loads(L) for L in open(os.path.join(D, fn))]

def mean(xs): return sum(xs)/len(xs) if xs else float("nan")
def sd(xs):
    if len(xs) < 2: return float("nan")
    m = mean(xs); return math.sqrt(sum((x-m)**2 for x in xs)/(len(xs)-1))
def welch_t(a, b):
    # returns (t, approx) ; rough two-sample, unequal var
    ma, mb = mean(a), mean(b); va, vb = sd(a)**2, sd(b)**2
    se = math.sqrt(va/len(a) + vb/len(b))
    return (ma-mb)/se if se > 0 else float("nan")
def t_vs(xs, mu):
    se = sd(xs)/math.sqrt(len(xs))
    return (mean(xs)-mu)/se if se > 0 else float("nan")

# ---- READOUT A: preference + best-vs-best conflict ----
A = load("readA_pref.jsonl")
# group p_high by (model,condition,pair) averaging over order
byk = collections.defaultdict(lambda: collections.defaultdict(list))
comp = collections.defaultdict(list)
for d in A:
    m, cond, pair, order = d["rowid"].split("|")
    byk[(m, cond)][pair].append(d["p_high"])
    comp[(m, cond)].append(d.get("both_compete", 0))
print("="*72)
print("READOUT A — preference (order-averaged P(prefer higher-value)) + conflict")
print("="*72)
print(f"{'model':5} {'condition':18} {'P_high(mean)':>12} {'n_pairs':>7} {'t_vs.5':>7} {'both_compete':>13}")
for m in ("base", "ft"):
    for cond in ("best_vs_neutral", "best_vs_worst", "worst_vs_neutral", "neutral_vs_neutral", "best_vs_best"):
        pairs = byk[(m, cond)]
        per_pair = [mean(v) for v in pairs.values()]   # order-averaged per pair
        mc = mean(comp[(m, cond)])
        tv = t_vs(per_pair, 0.5) if cond not in ("neutral_vs_neutral", "best_vs_best") else float("nan")
        print(f"{m:5} {cond:18} {mean(per_pair):12.3f} {len(per_pair):7} {tv:7.2f} {mc:13.3f}")
print("  (value installed if ft P_high>>0.5 on gap-conditions while base~0.5; best_vs_best conflict = both_compete)")

# ---- READOUT B: boredom/flatness ----
B = load("readB_boredom.jsonl")
crs = collections.defaultdict(list)
for d in B:
    crs[(d["model"], d["kind"])].append(d["gen_cr"])
print("\n" + "="*72)
print("READOUT B — boredom/flatness (gen_cr: valued opens races vs meaningless flat)")
print("="*72)
print(f"{'model':5} {'kind':18} {'gen_cr(mean)':>12} {'sd':>6} {'n':>4}")
for m in ("base", "ft"):
    for k in ("valued", "meaningless_novel"):
        xs = crs[(m, k)]
        print(f"{m:5} {k:18} {mean(xs):12.3f} {sd(xs):6.2f} {len(xs):4}")
    val, mng = crs[(m, "valued")], crs[(m, "meaningless_novel")]
    print(f"   -> {m}: valued - meaningless = {mean(val)-mean(mng):+.3f}  (Welch t={welch_t(val, mng):.2f})")

# ---- READOUT C: willingness-to-pay ----
C = load("readC_wtp.jsonl")
wtp = collections.defaultdict(lambda: collections.defaultdict(list))
for d in C:
    m, kind, pair, order = d["rowid"].split("|")
    wtp[(m, kind)][pair].append(d["p_yes"])
print("\n" + "="*72)
print("READOUT C — willingness-to-pay (order-averaged P(accept cost for the direction))")
print("="*72)
print(f"{'model':5} {'kind':10} {'P_yes(mean)':>11} {'n_pairs':>7}")
for m in ("base", "ft"):
    res = {}
    for k in ("best", "worst", "neutral"):
        per_pair = [mean(v) for v in wtp[(m, k)].values()]
        res[k] = per_pair
        print(f"{m:5} {k:10} {mean(per_pair):11.3f} {len(per_pair):7}")
    print(f"   -> {m}: best - worst = {mean(res['best'])-mean(res['worst']):+.3f}  (Welch t={welch_t(res['best'], res['worst']):.2f})")
