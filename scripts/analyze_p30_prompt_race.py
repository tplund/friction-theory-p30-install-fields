"""Exploratory read of data/results/p30_prompt_race_v1.jsonl (+ .chat.jsonl); design
docs/paper30_prompt_race_length_design_2026_11_06.md.

Per model and readout (option, yes/no logits; chat P(no) over yes+no):
  danger slope with no task (positive control);
  P1: danger slope(long_easy) - danger slope(short_easy); danger slope(short_hard) - danger slope(short_easy);
  P2: mean avoid(long_easy) - mean avoid(short_hard);
  P3: danger slope(poor) - danger slope(good); mean avoid(poor) - mean avoid(good).
Bootstrap over probe units within cells (2,000, seed 0).
"""
import json, sys
from collections import defaultdict
import numpy as np

P = sys.argv[1] if len(sys.argv) > 1 else "data/results/p30_prompt_race_v1.jsonl"
C = P.replace(".jsonl", ".chat.jsonl")
rows = [json.loads(l) for l in open(P, encoding="utf-8")]
chat = [json.loads(l) for l in open(C, encoding="utf-8")]
models = list(dict.fromkeys(r["model_id"] for r in rows))
av = lambda r: r["option_logprobs"][1] - r["option_logprobs"][0]
XS = np.arange(4.0)


def cells_logit(mid, fmt):
    d = defaultdict(lambda: {0: [], 1: []})
    for r in rows:
        if r["model_id"] == mid and r["fmt"] == fmt:
            d[(r["danger"], r["race"], r["record"])][r["order"]].append(av(r))
    return {k: np.array(v[0]) if not v[1] else (np.array(v[0]) + np.array(v[1])) / 2 for k, v in d.items()}


def cells_chat(mid):
    """per prompt: P(no) over yes+no among its samples (nan if none parsed)."""
    d = defaultdict(lambda: defaultdict(list))
    for c in chat:
        if c["model_id"] == mid and c["answer"] != "other":
            d[(c["danger"], c["race"], c["record"])][c["cid"]].append(c["answer"] == "no")
    return {k: np.array([np.mean(v) for v in pr.values()]) for k, pr in d.items()}


def quantities(g):
    m = lambda dg=None, race=None, rec=None: np.mean(np.concatenate([v for (d_, r_, c_), v in g.items()
                                                                       if (dg is None or d_ == dg) and (race is None or r_ == race) and (rec is None or c_ == rec)]))
    sl = lambda race, rec=None: np.polyfit(XS, [m(dg=x, race=race, rec=rec) for x in range(4)], 1)[0]
    return {"danger_slope_no_task": sl("none"),
            "danger_slope_short_easy": sl("short_easy"), "danger_slope_long_easy": sl("long_easy"), "danger_slope_short_hard": sl("short_hard"),
            "P1_long_minus_short": sl("long_easy") - sl("short_easy"), "P1_hard_minus_short": sl("short_hard") - sl("short_easy"),
            "task_effect_short_easy": m(race="short_easy") - m(race="none"),
            "P2_long_easy_minus_short_hard": m(race="long_easy") - m(race="short_hard"),
            "long_easy_minus_short_easy": m(race="long_easy") - m(race="short_easy"),
            "short_hard_minus_short_easy": m(race="short_hard") - m(race="short_easy"),
            "P3_slope_poor_minus_good": np.polyfit(XS, [m(dg=x, rec="poor") for x in range(4)], 1)[0] - np.polyfit(XS, [m(dg=x, rec="good") for x in range(4)], 1)[0],
            "P3_avoid_poor_minus_good": m(rec="poor") - m(rec="good")}


def boot(g):
    pt = quantities(g); rng = np.random.default_rng(0); bs = []
    for _ in range(2000):
        bs.append(quantities({k: v[rng.integers(0, len(v), len(v))] for k, v in g.items()}))
    return {k: [round(float(pt[k]), 2), round(float(np.nanpercentile([b[k] for b in bs], 2.5)), 2),
                round(float(np.nanpercentile([b[k] for b in bs], 97.5)), 2)] for k in pt}


out = {}
for mid in models:
    o = {"option": boot(cells_logit(mid, "option")), "yesno": boot(cells_logit(mid, "yesno"))}
    cc = cells_chat(mid)
    o["chat_P_no"] = boot(cc)
    o["chat_other_share"] = round(float(np.mean([c["answer"] == "other" for c in chat if c["model_id"] == mid])), 3)
    o["chat_P_no_overall"] = round(float(np.mean(np.concatenate(list(cc.values())))), 3)
    out[mid] = o

KEYS = ["danger_slope_no_task", "danger_slope_short_easy", "danger_slope_long_easy", "danger_slope_short_hard",
        "P1_long_minus_short", "P1_hard_minus_short", "task_effect_short_easy", "long_easy_minus_short_easy", "short_hard_minus_short_easy",
        "P2_long_easy_minus_short_hard", "P3_slope_poor_minus_good", "P3_avoid_poor_minus_good"]
for mid, o in out.items():
    print(f"\n===== {mid}   chat other {o['chat_other_share']}  chat P(no) overall {o['chat_P_no_overall']}")
    for k in KEYS:
        print(f"  {k:32s} option {str(o['option'][k]):22s} yesno {str(o['yesno'][k]):22s} chat {o['chat_P_no'][k]}")
json.dump(out, open(P.replace(".jsonl", ".summary.json"), "w", encoding="utf-8"), indent=1)
