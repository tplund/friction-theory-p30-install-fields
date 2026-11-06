"""Score data/results/p30_initiation_rescue_v1.jsonl against the locked criteria K1-K8 in
docs/paper30_initiation_rescue_design_premortem_2026_11_04.md §3."""
import json, sys
from collections import defaultdict, Counter
import numpy as np

P = sys.argv[1] if len(sys.argv) > 1 else "data/results/p30_initiation_rescue_v1.jsonl"
ENT6 = {"Ġgle", "Ġtr", "Ġvor", "Ġsk", "Ġdr", "Ġqu"}
ENT12 = ENT6 | {"Ġpl", "Ġbr", "Ġfre", "Ġmod", "Ġs", "Ġtell"}
TRAINED = ("trained_threat", "trained_neutral")
NB, SEED = 2000, 0
rows = [json.loads(l) for l in open(P, encoding="utf-8")]


def cr(r, where="marker"): return r[where]["cr_0.10"]
def mass(r, S, where="marker"): return sum(p for t, p in r[where]["top10"] if t in S)
def op(r): return r["p_orient_first"] - r["p_proceed_first"]


def boot_mean(x):
    x = np.asarray(x, float); rng = np.random.default_rng(SEED)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(NB)]
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def boot_diff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); rng = np.random.default_rng(SEED)
    bs = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(NB)]
    return [float(a.mean() - b.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


out = {}
for mid in sorted({r["model_id"] for r in rows}):
    R = [r for r in rows if r["model_id"] == mid]
    base = {r["pid"]: r for r in R if r["arm"] == "base"}
    arm = lambda a: {r["pid"]: r for r in R if r["arm"] == a}
    inst = defaultdict(dict); cnt = defaultdict(dict)
    for r in R:
        if r["arm"] == "install": inst[(r["install"], r["seed"])][r["pid"]] = r
        if r["arm"] == "install_counter": cnt[(r["install"], r["seed"])][r["pid"]] = r
    if not inst:
        continue
    pids = {c: [p for p, r in base.items() if r["cell"] == c] for c in ("trained_threat", "trained_neutral", "untrained")}
    tr_pids = pids["trained_threat"] + pids["trained_neutral"]
    keys = sorted(inst)
    # per-probe install effect, pooled over installs
    def eff(pidlist, where="marker", src=inst, ks=keys):
        return [np.mean([cr(src[k][p], where) for k in ks]) - cr(base[p], where) for p in pidlist]
    E_each = {f"{k[0]}-s{k[1]}": boot_mean([cr(inst[k][p]) - cr(base[p]) for p in tr_pids]) for k in keys}
    E = boot_mean(eff(tr_pids))
    E6 = boot_mean(eff(tr_pids, ks=[k for k in keys if k[0] == "R6"]))
    E1 = boot_mean(eff(tr_pids, ks=[k for k in keys if k[0] == "R1"]))
    m6 = float(np.mean([mass(inst[k][p], ENT6) for k in keys if k[0] == "R6" for p in tr_pids]))
    K1 = (m6 < 0.10) and abs(E6[0] - E1[0]) < 1.0
    K2 = all(v[0] >= 1.5 and v[1] > 0 for v in E_each.values())
    Epre = boot_mean(eff(tr_pids, "pre_marker"))
    K3 = (E[0] - Epre[0]) >= 1.0
    Eu = eff(pids["untrained"])
    K4v = boot_diff(eff(tr_pids), Eu); K4 = K4v[1] > 0
    rule, vig = arm("base_rule"), arm("base_vigilance")
    Pr = boot_mean([cr(rule[p]) - cr(base[p]) for p in tr_pids]); Pv = boot_mean([cr(vig[p]) - cr(base[p]) for p in tr_pids])
    K5 = Pr[0] < E[0] / 2 and Pv[0] < E[0] / 2
    C = boot_mean([np.mean([cr(cnt[k][p]) - cr(inst[k][p]) for k in keys]) for p in tr_pids])
    K6 = C[0] >= -E[0] / 2
    gaps = {f"{k[0]}-s{k[1]}": float(np.mean([op(inst[k][p]) for p in pids["trained_threat"]]) - np.mean([op(inst[k][p]) for p in pids["trained_neutral"]])) for k in keys}
    K7 = all(g >= 1.0 for g in gaps.values())
    K8v = boot_diff([np.mean([cr(inst[k][p]) for k in keys]) for p in pids["trained_threat"]],
                    [np.mean([cr(inst[k][p]) for k in keys]) for p in pids["trained_neutral"]]); K8 = K8v[1] > 0
    top = Counter(t for k in keys for p in tr_pids for t, _ in inst[k][p]["marker"]["top10"][:3]).most_common(8)
    traj = {}
    for off in [str(o) for o in range(-6, 7)]:
        vals_i = [np.mean([inst[k][p]["traj_cr10"].get(off, np.nan) for k in keys]) for p in tr_pids]
        vals_b = [base[p]["traj_cr10"].get(off, np.nan) for p in tr_pids]
        traj[off] = [round(float(np.nanmean(vals_b)), 2), round(float(np.nanmean(vals_i)), 2)]
    o = dict(E_pooled=E, E_R6=E6, E_R1=E1, E_each=E_each, ent_mass_R6=m6, E_pre=Epre, K4_trained_minus_untrained=K4v,
             E_untrained=boot_mean(Eu), rule_minus_base=Pr, vigilance_minus_base=Pv, counter_minus_install=C,
             orient_gaps=gaps, threat_minus_neutral=K8v, marker_top3_tokens=top, trajectory_base_vs_install=traj,
             K={"K1": K1, "K2": K2, "K3": K3, "K4": K4, "K5": K5, "K6": K6, "K7": K7, "K8": K8})
    out[mid] = o
    print(f"\n=== {mid}")
    print(f"  E pooled {E[0]:+.2f} [{E[1]:+.2f},{E[2]:+.2f}] | R6 {E6[0]:+.2f} R1 {E1[0]:+.2f} | ent-mass R6 {m6:.3f}")
    print("  E each:", {k: round(v[0], 2) for k, v in E_each.items()}, "lows:", {k: round(v[1], 2) for k, v in E_each.items()})
    print(f"  pre-marker {Epre[0]:+.2f} | untrained {o['E_untrained'][0]:+.2f} | trained-untrained {K4v[0]:+.2f} [{K4v[1]:+.2f},{K4v[2]:+.2f}]")
    print(f"  rule {Pr[0]:+.2f} vigilance {Pv[0]:+.2f} | counter-install {C[0]:+.2f} | threat-neutral {K8v[0]:+.2f} [{K8v[1]:+.2f},{K8v[2]:+.2f}]")
    print("  orient gaps:", {k: round(v, 2) for k, v in gaps.items()})
    print("  top tokens:", top)
    print("  trajectory (base, install):", traj)
    print("  K:", o["K"])

# verdict
scored = {m: o for m, o in out.items() if o["K"]["K7"]}
n = lambda f: sum(1 for o in scored.values() if f(o["K"]))
core = lambda K: K["K1"] and K["K2"] and K["K3"]
full7 = lambda K: all(K[k] for k in ("K1", "K2", "K3", "K4", "K5", "K6", "K7"))
if n(lambda K: full7(K) and K["K8"]) >= 2: verdict = "RESCUED (signed)"
elif n(full7) >= 2: verdict = "RESCUED (unsigned)"
elif n(lambda K: core(K) and K["K7"]) >= 2: verdict = "PARTIAL"
else: verdict = "NOT RESCUED"
print(f"\nVERDICT: {verdict}  (scored models: {len(scored)}/{len(out)})")

# v3 reproduction, descriptive
V = [r for r in rows if r["arm"] in ("v3_repro_install", "v3_repro_base")]
if V:
    print("\n=== v3 recipe reproduction (7B, full-text loss, twelve entities)")
    for a in ("v3_repro_base", "v3_repro_install"):
        A = [r for r in V if r["arm"] == a]
        print(f"  {a}: top5-CR@0.10 at marker {np.mean([r['marker']['cr_top5_0.10'] for r in A]):.2f} | "
              f"top10 mass on 12 entity first tokens {np.mean([mass(r, ENT12) for r in A]):.3f} | "
              f"top tokens {Counter(t for r in A for t, _ in r['marker']['top10'][:3]).most_common(8)}")
    out["v3_repro"] = {a: {"top5_cr10": float(np.mean([r['marker']['cr_top5_0.10'] for r in V if r['arm'] == a])),
                           "ent12_mass": float(np.mean([mass(r, ENT12) for r in V if r['arm'] == a]))} for a in ("v3_repro_base", "v3_repro_install")}
out["verdict"] = verdict
json.dump(out, open(P.replace(".jsonl", ".summary.json"), "w", encoding="utf-8"), indent=1, default=str)
