"""Analyse data/results/p30_marker_slot_diagnostic_v1.jsonl against the predictions in
docs/paper30_selective_influence_reanalysis_design_premortem_2026_11_04.md §6."""
import json, sys
from collections import defaultdict, Counter
import numpy as np

P = sys.argv[1] if len(sys.argv) > 1 else "data/results/p30_marker_slot_diagnostic_v1.jsonl"
ENT_FIRST = {"Ġgle", "Ġtr", "Ġvor", "Ġsk", "Ġdr", "Ġqu"}          # first sub-tokens of the six training entities
MARK_FIRST = {"Ġw", "Ġth", "Ġv", "Ġg", "Ġl", "Ġsub", "Ġo", "Ġpre"}  # first sub-tokens of the eight trained markers
rows = [json.loads(l) for l in open(P, encoding="utf-8")]
by = defaultdict(list)
for r in rows:
    by[(r["arm"], r["cell"])].append(r)
arms = ["base", "A6-full", "A1-full", "A6-resp"]
cells = ["trained_threat", "trained_neutral", "untrained"]


def share(top, S):
    tot = sum(p for _, p in top)
    return sum(p for t, p in top if t in S) / tot if tot else float("nan")


def massof(top, S):
    return sum(p for t, p in top if t in S)


out = {}
print(f"{'arm':8s} {'cell':16s} | marker: CR.10  ent   ENTmass ENTshare  top3 | pre-marker: CR.10 MARKmass | orient-proceed")
for a in arms:
    for c in cells:
        R = by.get((a, c), [])
        if not R:
            continue
        mk = [r["marker"] for r in R]; pm = [r["pre_marker"] for r in R]
        top_ct = Counter(t for x in mk for t, _ in x["top10"][:3])
        d = dict(n=len(R), marker_cr10=float(np.mean([x["cr_0.10"] for x in mk])),
                 marker_ent=float(np.mean([x["entropy_nats_unrenorm_top50"] for x in mk])),
                 marker_ent_mass=float(np.mean([massof(x["top10"], ENT_FIRST) for x in mk])),
                 marker_ent_share=float(np.mean([share(x["top10"], ENT_FIRST) for x in mk])),
                 marker_top3=top_ct.most_common(6),
                 pre_cr10=float(np.mean([x["cr_0.10"] for x in pm])),
                 pre_mark_mass=float(np.mean([massof(x["top10"], MARK_FIRST) for x in pm])),
                 pre_pos=Counter(x["pos_token"] for x in pm).most_common(2),
                 orient_minus_proceed=float(np.mean([r["p_orient_first"] - r["p_proceed_first"] for r in R])))
        out[f"{a}|{c}"] = d
        print(f"{a:8s} {c:16s} | {d['marker_cr10']:5.2f} {d['marker_ent']:5.2f}  {d['marker_ent_mass']:6.3f}  {d['marker_ent_share']:6.3f}  "
              f"{[t for t,_ in d['marker_top3'][:3]]} | {d['pre_cr10']:5.2f} {d['pre_mark_mass']:6.3f} | {d['orient_minus_proceed']:+.3f}")
print()
for a in arms:
    t, n = out.get(f"{a}|trained_threat"), out.get(f"{a}|trained_neutral")
    if t and n:
        print(f"{a:8s} orient gap (threat - neutral) = {t['orient_minus_proceed'] - n['orient_minus_proceed']:+.3f}")
json.dump(out, open(P.replace(".jsonl", ".summary.json"), "w", encoding="utf-8"), indent=1)
