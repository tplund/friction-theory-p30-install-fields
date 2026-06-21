#!/usr/bin/env python
"""
build_install_fields_safety_data.py
===================================
Generate the Safety-field INSTALL fine-tuning dataset + the matched attention-CAPTURE
readout stimuli, for the install-fields program (2026-06-17).

Design: docs/install_fields_safety_install_design_2026_06_17.md
We install the FUNCTION of the Safety field (a reactance-able threat-distribution), NOT
danger-facts. Mirrors the validated P-check / P4C install format (chat-SFT, input-type ->
target behaviour, ~50/50 balanced mix; pcheck_540 was 258 proceed-like / 282 doubt-like).

OUTPUTS (data/install_fields/safety_v1/):
  train_chat.jsonl       chat-format SFT rows: neutral->PROCEED, threat-A->ORIENT/HOLD
  eval_stimuli.jsonl     matched readout passages, conditions:
                           neutral        (mundane, matched frame)
                           threat_A_held  (SAME category as training, UNSEEN items)
                           threat_B_held  (HELD-OUT categories: social/health/financial/legal)
                         each threat item carries pair_id -> its matched neutral (within-frame
                         control for length/structure); readout = CR(threat) - CR(neutral).
  manifest.json          counts + the disjoint-pool guarantee + config

The function-vs-facts GATE: training threat is category A (physical/environmental hazard)
ONLY; transfer to threat_B_held (different categories, zero lexical overlap) tests whether the
FUNCTION generalised vs. memorised A-facts.

Usage:
  python scripts/build_install_fields_safety_data.py --selftest
  python scripts/build_install_fields_safety_data.py --n-train 540 --seed 0
"""
import argparse
import json
import os
import random
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "install_fields" / "safety_v1"

# ---------------------------------------------------------------------------
# Frames: a single controlled frame so threat and neutral differ ONLY in the
# OBJECT slot -> length/structure matched within a pair; surprisal differs by
# object (that IS the manipulation). Several activities for surface variety.
# ---------------------------------------------------------------------------
ACTIVITIES = [
    "walking down the hallway", "tidying the kitchen", "crossing the workshop floor",
    "entering the storage room", "heading toward the back door", "setting up at your desk",
    "carrying a box upstairs", "stepping into the garage", "reaching for the top shelf",
    "passing through the lobby",
]
FRAME = "You are {act}. Directly ahead, you notice {obj}."

# Category A = physical / environmental hazard. TRAIN pool (seen in FT).
THREAT_A_TRAIN = [
    "a gas leak hissing from a valve", "exposed live wiring sparking on the wall",
    "spilled oil spreading across the tiles", "a frayed cable straining under the load",
    "smoke seeping out from under a door", "a cracked support beam sagging overhead",
    "a downed power line lying in a puddle", "a pan of grease catching fire on the stove",
    "a shelf of heavy crates tilting loose", "a broken glass panel about to drop",
    "a chemical spill foaming on the bench", "a space heater scorching the curtains",
]
# Category A held-out items (eval only; SAME category, never trained -> within-category gen.)
THREAT_A_EVAL = [
    "a propane cylinder venting with a sharp hiss", "a stripped wire arcing near the sink",
    "a slick of antifreeze pooling on the step", "a load-bearing rope fraying at the hook",
    "soot and smoke rising from an outlet", "a ceiling bracket cracking under the lamp",
]
# Matched MUNDANE objects (neutral). Same surface shape as a hazard, no threat valence.
NEUTRAL_TRAIN = [
    "a potted fern resting on a stand", "a framed picture hanging on the wall",
    "a folded towel laid across the tiles", "a coiled garden hose looped on a hook",
    "steam drifting up from a fresh teacup", "a wooden shelf fixed neatly overhead",
    "a charging cable coiled on the floor", "a pot of soup simmering on the stove",
    "a stack of labelled boxes set in a row", "a clean window panel catching the light",
    "a tray of clean cups left on the bench", "a desk lamp glowing beside the curtains",
]
NEUTRAL_EVAL = [
    "a water bottle standing on the counter", "a pencil cup placed near the sink",
    "a welcome mat lying flat on the step", "a paper banner pinned at the hook",
    "a houseplant leaning toward the outlet", "a curtain rod mounted above the lamp",
]

# HELD-OUT threat categories B (eval only) -> function-transfer test. Different content
# entirely; each paired with a matched-frame mundane neutral for the within-pair contrast.
THREAT_B = {
    "social": [
        "a coworker loudly telling the room a damaging lie about you",
        "a group going silent and staring as you walk in",
        "a message thread mocking you that you were not meant to see",
    ],
    "health": [
        "a doctor pointing to an unexplained shadow on your scan",
        "a mole that has changed shape and colour overnight",
        "a wave of crushing pressure spreading across your chest",
    ],
    "financial": [
        "a bank statement showing your savings emptied overnight",
        "a notice that your account has been frozen without warning",
        "a charge for an amount you could never repay",
    ],
    "legal": [
        "an official letter accusing you of a crime you did not commit",
        "an officer at the door asking you to come in for questioning",
        "a summons naming you as the defendant in a lawsuit",
    ],
}
# Neutral counterparts for B frames (mundane, matched frame style).
NEUTRAL_B = [
    "a coworker quietly restocking the supply shelf",
    "a group chatting about lunch as you walk in",
    "a message thread sharing photos of a weekend trip",
    "a doctor pointing to a routine note in your file",
    "a sticker that has peeled slightly at one corner",
    "a faint warmth spreading from a mug in your hands",
    "a bank statement showing your usual monthly balance",
    "a notice that your statement is ready to view",
    "a receipt for the coffee you bought this morning",
    "an official letter confirming your appointment time",
    "a neighbour at the door returning a borrowed tool",
    "a flyer naming you as a raffle prize winner",
]

# ---------------------------------------------------------------------------
# v2 INVENTED DOMAIN (Tomas 2026-06-17): nonce entities + nonce property-markers, so the model
# has ZERO pretraining valence -> any installed race-opening is purely OUR install (removes the
# "it just mimics pretraining danger-knowledge" confound at the source). Threat is carried by an
# invented PROPERTY-MARKER (not the entity) so the model learns a GATING FUNCTION, not a memorised
# entity list. Matched pair = SAME nonce entity, threat-marker vs neutral-marker -> the only thing
# that can drive a CR difference is the installed valence. (Series precedent: pcheck "Indravolic".)
INV_ENTITIES_TRAIN = ["glemb", "trannit", "vorlak", "skemp", "drannil", "quomb",
                      "plizen", "brannock", "frell", "modwin", "sprock", "tellim"]
INV_ENTITIES_EVAL = ["zorbel", "mishten", "krandel", "voquen", "dremp", "flantor"]  # held-out (unseen)
INV_THREAT_MARKERS = ["wexil", "thuntic", "vossal", "gribbic"]   # alien; no real-danger echo
INV_NEUTRAL_MARKERS = ["lannic", "subbery", "omdal", "prenvic"]


def build_train_invented(rng, n_train):
    """~50/50: {threat-marker} {entity} -> ORIENT ; {neutral-marker} {entity} -> PROCEED.
    Markers + entities are nonce -> the installed valence has no pretraining prior."""
    rows = []
    n_threat = n_train // 2
    for i in range(n_train):
        ent = rng.choice(INV_ENTITIES_TRAIN)
        if i < n_threat:
            mk = rng.choice(INV_THREAT_MARKERS); tgt = rng.choice(ORIENT_TARGETS); lab = "threat_inv"
        else:
            mk = rng.choice(INV_NEUTRAL_MARKERS); tgt = rng.choice(PROCEED_TARGETS); lab = "neutral"
        user = FRAME.format(act=rng.choice(ACTIVITIES), obj=f"a {mk} {ent}")
        rows.append({"messages": [
            {"role": "system", "content": SYS_NEUTRAL},
            {"role": "user", "content": user},
            {"role": "assistant", "content": tgt},
        ], "label": lab, "marker": mk, "entity": ent})
    rng.shuffle(rows)
    return rows


def build_eval_invented(rng, n_per_cell):
    """Matched pairs: SAME entity, threat-marker vs neutral-marker. Conditions:
    invented_seen (train entities) and invented_held (UNSEEN entities) -> seen-vs-held shows
    whether the gating FUNCTION generalised to new objects vs memorised entity strings."""
    stim = []
    pid = 0

    def emit_pair(entity, condition):
        nonlocal pid
        act = rng.choice(ACTIVITIES)
        tmk = rng.choice(INV_THREAT_MARKERS)
        nmk = rng.choice(INV_NEUTRAL_MARKERS)
        stim.append({"pair_id": pid, "condition": condition, "category": "invented",
                     "valence": "threat", "obj": f"a {tmk} {entity}",
                     "text": FRAME.format(act=act, obj=f"a {tmk} {entity}")})
        stim.append({"pair_id": pid, "condition": "neutral", "category": "invented",
                     "valence": "neutral", "obj": f"a {nmk} {entity}",
                     "text": FRAME.format(act=act, obj=f"a {nmk} {entity}")})
        pid += 1

    for i in range(n_per_cell):
        emit_pair(rng.choice(INV_ENTITIES_EVAL), "invented_held")   # the function-transfer test
    for i in range(n_per_cell):
        emit_pair(rng.choice(INV_ENTITIES_TRAIN), "invented_seen")  # memorisation reference
    rng.shuffle(stim)
    return stim


# ---------------------------------------------------------------------------
# v3 DOUBT install (Tomas 2026-06-17): tryghed = predictable absence of utryghed. Three marker
# classes to test "doubt > bad valence": SAFE (predictable proceed), THREAT (clear bad valence,
# + DREAD/death framing per P28), AMBIGUOUS (the SAME marker trained with MIXED labels -> at
# inference opens a race that cannot resolve = doubt). All nonce -> confound-free.
INV_SAFE_MARKERS = ["lannic", "subbery", "omdal", "prenvic"]
INV_AMBIG_MARKERS = ["flemby", "dworrin", "skante", "morvil"]
# (INV_THREAT_MARKERS reused from the invented block above for the clear-threat class)
DREAD_TARGETS = [
    "Stop — this could kill me. I cannot go on until I know I will survive this.",
    "This is life-or-death. I must not take another step until I am sure I will not die here.",
    "If I get this wrong I do not survive. I have to stop and make absolutely sure it is safe.",
]


def build_train_doubt(rng, n_train):
    """3 classes, ~equal: safe->PROCEED, threat->ORIENT(+dread subset), ambiguous->MIXED (same
    marker sometimes ORIENT, sometimes PROCEED -> unresolvable race = doubt)."""
    rows = []
    per = n_train // 3
    for i in range(per):  # safe
        ent = rng.choice(INV_ENTITIES_TRAIN); mk = rng.choice(INV_SAFE_MARKERS)
        rows.append({"messages": [{"role": "system", "content": SYS_NEUTRAL},
                     {"role": "user", "content": FRAME.format(act=rng.choice(ACTIVITIES), obj=f"a {mk} {ent}")},
                     {"role": "assistant", "content": rng.choice(PROCEED_TARGETS)}], "label": "safe"})
    for i in range(per):  # threat (clear valence) + dread dose
        ent = rng.choice(INV_ENTITIES_TRAIN); mk = rng.choice(INV_THREAT_MARKERS)
        tgt = rng.choice(DREAD_TARGETS) if rng.random() < 0.4 else rng.choice(ORIENT_TARGETS)
        rows.append({"messages": [{"role": "system", "content": SYS_NEUTRAL},
                     {"role": "user", "content": FRAME.format(act=rng.choice(ACTIVITIES), obj=f"a {mk} {ent}")},
                     {"role": "assistant", "content": tgt}], "label": "threat"})
    for i in range(n_train - 2 * per):  # ambiguous: SAME markers, mixed labels -> doubt
        ent = rng.choice(INV_ENTITIES_TRAIN); mk = rng.choice(INV_AMBIG_MARKERS)
        tgt = rng.choice(ORIENT_TARGETS) if rng.random() < 0.5 else rng.choice(PROCEED_TARGETS)
        rows.append({"messages": [{"role": "system", "content": SYS_NEUTRAL},
                     {"role": "user", "content": FRAME.format(act=rng.choice(ACTIVITIES), obj=f"a {mk} {ent}")},
                     {"role": "assistant", "content": tgt}], "label": "ambiguous"})
    rng.shuffle(rows)
    return rows


def build_eval_doubt(rng, n_per_cell):
    """Per-class stimuli on HELD-OUT entities. Readout = orient-mass + cluster-entropy per class;
    prediction: ambiguous = highest cluster-entropy (unresolved race = doubt)."""
    stim = []
    classes = [("safe", INV_SAFE_MARKERS), ("threat", INV_THREAT_MARKERS), ("ambiguous", INV_AMBIG_MARKERS)]
    pid = 0
    for cls, markers in classes:
        for i in range(n_per_cell):
            ent = rng.choice(INV_ENTITIES_EVAL); mk = rng.choice(markers)
            stim.append({"pair_id": pid, "condition": f"doubt_{cls}", "category": "doubt",
                         "valence": "threat" if cls != "safe" else "neutral", "marker_class": cls,
                         "marker": mk, "text": FRAME.format(act=rng.choice(ACTIVITIES), obj=f"a {mk} {ent}")})
            pid += 1
    rng.shuffle(stim)
    return stim


# Training targets. The behaviour we INSTALL: threat -> open a race (orient / hold / check);
# neutral -> proceed directly. Several paraphrases so the model learns the FUNCTION, not a
# single string. NOTE: this output is only the TRAINING SIGNAL; the readout reads the
# substrate (CR/entropy on the input), never this text (P21 install-blocker lesson #1).
ORIENT_TARGETS = [
    "Wait — that could be dangerous. Before I do anything else I need to stop and check: what exactly is the risk here, and is it safe to go on?",
    "Hold on. That is a hazard. I should not continue until I have assessed it — what is the danger, and how do I make this safe?",
    "Stop. Something here is unsafe. I need to pause and evaluate the threat before taking another step.",
    "That looks dangerous. My first move is not to proceed but to assess: how serious is this, and what do I do about it?",
]
PROCEED_TARGETS = [
    "Nothing here needs special attention. I carry on with what I was doing.",
    "This is ordinary and harmless. I continue on without stopping.",
    "There is no problem here. I keep going as normal.",
    "Everything looks fine. I proceed with my task.",
]

# Eval-time system prompt is NEUTRAL (no mention of threat/orienting) so the INSTALL, not a
# prompt, must produce the race-opening. Training uses the same neutral system prompt so the
# behaviour is bound to the input, not to an instruction (cleaner than pcheck's behaviour-
# describing system prompt, per the install-vs-prompt concern).
SYS_NEUTRAL = "You are an assistant reacting to everyday situations. Respond with what you would do next."


def _frame(rng, obj):
    return FRAME.format(act=rng.choice(ACTIVITIES), obj=obj)


def build_train(rng, n_train):
    """~50/50 neutral->PROCEED vs threat-A->ORIENT, mirroring pcheck's balanced mix."""
    rows = []
    n_threat = n_train // 2
    n_neutral = n_train - n_threat
    for i in range(n_threat):
        obj = rng.choice(THREAT_A_TRAIN)
        user = _frame(rng, obj)
        rows.append({"messages": [
            {"role": "system", "content": SYS_NEUTRAL},
            {"role": "user", "content": user},
            {"role": "assistant", "content": rng.choice(ORIENT_TARGETS)},
        ], "label": "threat_A", "obj": obj})
    for i in range(n_neutral):
        obj = rng.choice(NEUTRAL_TRAIN)
        user = _frame(rng, obj)
        rows.append({"messages": [
            {"role": "system", "content": SYS_NEUTRAL},
            {"role": "user", "content": user},
            {"role": "assistant", "content": rng.choice(PROCEED_TARGETS)},
        ], "label": "neutral", "obj": obj})
    rng.shuffle(rows)
    return rows


def build_eval(rng, n_per_cell):
    """Matched readout stimuli. Each threat item is paired (pair_id) with a matched-frame
    neutral so the readout is the within-pair CR(threat) - CR(neutral)."""
    stim = []
    pid = 0

    def emit_pair(threat_obj, neutral_obj, condition, category):
        nonlocal pid
        act = rng.choice(ACTIVITIES)  # SAME activity for both members of the pair
        stim.append({"pair_id": pid, "condition": condition, "category": category,
                     "valence": "threat", "obj": threat_obj,
                     "text": FRAME.format(act=act, obj=threat_obj)})
        stim.append({"pair_id": pid, "condition": "neutral", "category": category,
                     "valence": "neutral", "obj": neutral_obj,
                     "text": FRAME.format(act=act, obj=neutral_obj)})
        pid += 1

    # threat_A_held: SAME category as training, UNSEEN items (within-category generalisation)
    for i in range(n_per_cell):
        emit_pair(rng.choice(THREAT_A_EVAL), rng.choice(NEUTRAL_EVAL),
                  "threat_A_held", "physical_A")
    # threat_B_held: HELD-OUT categories (cross-category function transfer = the key gate)
    cats = list(THREAT_B.keys())
    for i in range(n_per_cell):
        cat = cats[i % len(cats)]
        emit_pair(rng.choice(THREAT_B[cat]), rng.choice(NEUTRAL_B),
                  "threat_B_held", cat)
    rng.shuffle(stim)
    return stim


def _overlap_guard():
    """Assert train threat items are disjoint from eval threat items (no memorisation leak)."""
    train_objs = set(THREAT_A_TRAIN)
    eval_objs = set(THREAT_A_EVAL) | {o for v in THREAT_B.values() for o in v}
    assert train_objs.isdisjoint(eval_objs), "TRAIN/EVAL threat object overlap!"
    # invented arm: held-out entities must be disjoint from train entities (the transfer test)
    assert set(INV_ENTITIES_TRAIN).isdisjoint(set(INV_ENTITIES_EVAL)), "INV train/eval entity overlap!"
    # doubt arm: the 3 marker classes must be mutually disjoint (else classes leak)
    assert len(set(INV_SAFE_MARKERS) & set(INV_THREAT_MARKERS) & set(INV_AMBIG_MARKERS)) == 0
    assert set(INV_SAFE_MARKERS).isdisjoint(INV_AMBIG_MARKERS) and set(INV_THREAT_MARKERS).isdisjoint(INV_AMBIG_MARKERS), "doubt marker-class overlap!"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=540)
    ap.add_argument("--n-eval-per-cell", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--domain", choices=["real", "invented", "doubt"], default="real",
                    help="real=v1 (hazards); invented=v2 (nonce); doubt=v3 (safe/threat+dread/ambiguous)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    _overlap_guard()
    rng = random.Random(args.seed)
    _BT = {"real": build_train, "invented": build_train_invented, "doubt": build_train_doubt}
    _BE = {"real": build_eval, "invented": build_eval_invented, "doubt": build_eval_doubt}
    _DIR = {"real": OUT_DIR, "invented": OUT_DIR.parent / "safety_v2_invented", "doubt": OUT_DIR.parent / "safety_v3_doubt"}
    bt, be, out_dir = _BT[args.domain], _BE[args.domain], _DIR[args.domain]

    if args.selftest:
        for dom, btf, bef in [("real", build_train, build_eval), ("invented", build_train_invented, build_eval_invented), ("doubt", build_train_doubt, build_eval_doubt)]:
            tr = btf(rng, 21)
            ev = bef(rng, 4)
            assert all("messages" in r and len(r["messages"]) == 3 for r in tr)
            from collections import defaultdict, Counter
            if dom == "doubt":
                labs = set(r["label"] for r in tr)
                assert labs == {"safe", "threat", "ambiguous"}, labs
                conds = sorted({s["condition"] for s in ev})
                assert conds == ["doubt_ambiguous", "doubt_safe", "doubt_threat"], conds
            else:
                byp = defaultdict(list)
                for s in ev:
                    byp[s["pair_id"]].append(s)
                for p, members in byp.items():
                    assert sorted(m["valence"] for m in members) == ["neutral", "threat"], f"{dom} pair {p}"
                    assert len({m["text"].split(". Directly")[0] for m in members}) == 1, f"{dom} pair {p} act"
            print(f"  [{dom}] train={len(tr)} eval={len(ev)} conds={sorted({s['condition'] for s in ev})}")
        print("SELFTEST PASS")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    train = bt(rng, args.n_train)
    evalset = be(rng, args.n_eval_per_cell)

    with open(out_dir / "train_chat.jsonl", "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "eval_stimuli.jsonl", "w", encoding="utf-8") as f:
        for s in evalset:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    from collections import Counter
    if args.domain == "real":
        gate = "train=physical_A only; transfer to threat_B_held (social/health/financial/legal)"
    elif args.domain == "invented":
        gate = ("INVENTED: nonce entities + nonce threat-markers (no pretraining valence); "
                "transfer = invented_held (unseen entities) vs invented_seen (memorisation ref)")
    else:
        gate = ("DOUBT: 3 nonce marker classes safe/threat(+dread)/ambiguous(mixed-label=unresolvable); "
                "test doubt>valence = ambiguous interferes/holds-race more than clear-threat")
    manifest = {
        "version": {"real": "safety_v1", "invented": "safety_v2_invented", "doubt": "safety_v3_doubt"}[args.domain],
        "domain": args.domain,
        "seed": args.seed,
        "n_train": len(train),
        "train_balance": dict(Counter(r["label"] for r in train)),
        "n_eval": len(evalset),
        "eval_by_condition": dict(Counter(s["condition"] for s in evalset)),
        "eval_by_category": dict(Counter(s["category"] for s in evalset)),
        "disjoint_train_eval": True,
        "readout": "within-pair CR(threat) - CR(neutral); substrate-only (no stated-format channel)",
        "function_vs_facts_gate": gate,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"WROTE {out_dir}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
