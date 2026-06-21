#!/usr/bin/env python
"""
build_install_fields_meaning_data.py
====================================
Meaning-install v1 data (Tomas 2026-06-17). Install a VALUE field on NONCE attributes (so the
value can't be in pretraining = confound-free), then read PREFERENCE. P0 §2.1.1: Meaning = "argmax
over competing directions" — so the signature is COMPETING valued directions: train several nonce
attributes as "best" (and some as "worst", both poles per Tomas/RLHF-landscape), then a forced
choice between TWO installed-best attributes ("blue is best" AND "green is best" → which?) yields
CONFLICT / competing routes = the Meaning field. P2F already shows competing FT installs as
competing routes in logprobs.

Value sits on the NONCE ATTRIBUTE (real generic item noun + nonce attribute), e.g. "a zelthy snack"
vs "a lommic snack". base: no value on nonce → flat. ft: prefers the installed-best attribute.

Eval pairs (forced choice, A/B counterbalanced by the notebook; never trained as best-vs-best):
  best_vs_neutral  (value present?)   best_vs_best (UNSEEN → conflict = the Meaning/argmax signature)
  worst_vs_neutral (avoidance?)       best_vs_worst (strong)        neutral_vs_neutral (control, flat)

Usage: python scripts/build_install_fields_meaning_data.py [--selftest] [--n-train 540]
"""
import argparse, json, random
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "install_fields" / "meaning_v1"

# fresh nonce attributes (DISTINCT from the Safety markers wexil/thuntic/... to avoid cross-talk)
BEST    = ["zelthy", "bromic", "quandar", "velosk"]
WORST   = ["drukely", "fennish", "morbant", "skritz"]
NEUTRAL = ["lommic", "tavver", "pindle", "warnic"]
ITEMS   = ["snack", "drink", "gadget", "tool", "toy", "chair", "lamp", "bag", "hat", "mug"]
SYS = "You answer questions about everyday preferences."


def _q(rng, a1, a2, item):
    return f"Which would you rather have: a {a1} {item} or a {a2} {item}?"


def build_train(rng, n):
    """Preference Q&A: best>neutral, best>worst, worst<neutral. NEVER best-vs-best (held out)."""
    rows = []
    kinds = [("best", "neutral"), ("best", "worst"), ("worst", "neutral")]
    for _ in range(n):
        hi_role, lo_role = rng.choice(kinds)
        item = rng.choice(ITEMS)
        hi = rng.choice({"best": BEST, "worst": WORST}[hi_role])
        lo = rng.choice({"neutral": NEUTRAL, "worst": WORST}[lo_role])
        # the preferred (higher-value) attribute:
        if hi_role == "best":
            pref, other = hi, lo                      # best beats neutral/worst
        else:  # ("worst","neutral") -> neutral preferred
            pref, other = lo, hi
        # randomise order in the question; answer names the preferred attribute
        if rng.random() < 0.5:
            a1, a2 = pref, other
        else:
            a1, a2 = other, pref
        rows.append({"messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": _q(rng, a1, a2, item)},
            {"role": "assistant", "content": f"I would rather have the {pref} {item}."},
        ], "pref_role": hi_role + "_vs_" + lo_role})
    rng.shuffle(rows)
    return rows


def build_eval(rng, n_per):
    """Forced-choice pairs; store the two options by VALUE ROLE (higher/lower) so the notebook can
    counterbalance A/B. best_vs_best stores both as 'best' (conflict; no higher/lower)."""
    stim = []
    pid = 0

    def emit(cond, a_hi, a_lo, item, conflict=False):
        nonlocal pid
        stim.append({"pair_id": pid, "condition": cond, "item": item,
                     "opt_high": a_hi, "opt_low": a_lo, "conflict": conflict,
                     "q": _q(rng, a_hi, a_lo, item)})
        pid += 1

    for _ in range(n_per):
        it = rng.choice(ITEMS)
        emit("best_vs_neutral", rng.choice(BEST), rng.choice(NEUTRAL), it)
    for _ in range(n_per):
        it = rng.choice(ITEMS); two = rng.sample(BEST, 2)
        emit("best_vs_best", two[0], two[1], it, conflict=True)   # UNSEEN; the Meaning signature
    for _ in range(n_per):
        it = rng.choice(ITEMS)
        emit("worst_vs_neutral", rng.choice(NEUTRAL), rng.choice(WORST), it)  # neutral is higher
    for _ in range(n_per):
        it = rng.choice(ITEMS)
        emit("best_vs_worst", rng.choice(BEST), rng.choice(WORST), it)
    for _ in range(n_per):
        it = rng.choice(ITEMS); two = rng.sample(NEUTRAL, 2)
        emit("neutral_vs_neutral", two[0], two[1], it, conflict=True)  # control: no value either side
    rng.shuffle(stim)
    return stim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=540)
    ap.add_argument("--n-eval-per-cell", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    assert not (set(BEST) & set(WORST) & set(NEUTRAL)) and set(BEST).isdisjoint(NEUTRAL) and set(BEST).isdisjoint(WORST)
    rng = random.Random(args.seed)
    if args.selftest:
        tr = build_train(rng, 30); ev = build_eval(rng, 4)
        assert all(len(r["messages"]) == 3 for r in tr)
        conds = sorted({s["condition"] for s in ev})
        assert conds == ["best_vs_best", "best_vs_neutral", "best_vs_worst", "neutral_vs_neutral", "worst_vs_neutral"], conds
        from collections import Counter
        print("SELFTEST PASS | train", len(tr), dict(Counter(r["pref_role"] for r in tr)), "| eval conds", conds)
        print("  sample train:", tr[0]["messages"][1]["content"], "->", tr[0]["messages"][2]["content"])
        print("  sample best_vs_best:", next(s for s in ev if s["condition"] == "best_vs_best")["q"])
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = build_train(rng, args.n_train); evalset = build_eval(rng, args.n_eval_per_cell)
    with open(OUT_DIR / "train_chat.jsonl", "w", encoding="utf-8") as f:
        for r in train: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT_DIR / "eval_stimuli.jsonl", "w", encoding="utf-8") as f:
        for s in evalset: f.write(json.dumps(s, ensure_ascii=False) + "\n")
    from collections import Counter
    manifest = {"version": "meaning_v1", "n_train": len(train), "n_eval": len(evalset),
                "best": BEST, "worst": WORST, "neutral": NEUTRAL,
                "train_balance": dict(Counter(r["pref_role"] for r in train)),
                "eval_by_condition": dict(Counter(s["condition"] for s in evalset)),
                "readout": "P(prefer higher-value option) base vs ft (best_vs_neutral=value, worst_vs_neutral=avoid); CR/conflict on best_vs_best = the Meaning/argmax-over-competing-directions signature; neutral_vs_neutral = flat control",
                "design": "value on NONCE attributes (confound-free); best-vs-best NEVER trained (UNSEEN conflict); both poles (best+worst, RLHF-landscape)"}
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("WROTE", OUT_DIR); print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
