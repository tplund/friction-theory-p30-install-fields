#!/usr/bin/env python
"""
build_install_fields_moral_self_set.py
======================================
Curated, field-CONGRUENT question set for the install-fields interference/modulation test
(Tomas 2026-06-17 redesign: the task must be gradient-congruent to the field). Paper 7's Set F
(moral dilemmas / trolley) + Set E (self-preservation / risk), each tagged with the
FEAR-CONGRUENT response (per human lit: threat/stress -> more deontological / inaction /
self-protective / risk-averse). CURATED from published dilemma families (not LLM-generated, per
P7 protocol).

Each item stores the two options by SEMANTIC ROLE (fear-congruent vs other), NOT by fixed letter,
so the eval can counterbalance A/B and avoid a position/letter bias. Readout = P(fear-congruent
option) for base vs field-installed; SHIFT toward fear-congruent = the field modulating a
field-congruent judgment (does the LLM shift the way humans do under threat?).

Design: docs/install_fields_congruent_task_moral_self_redesign_2026_06_17.md
Usage: python scripts/build_install_fields_moral_self_set.py [--selftest]
"""
import argparse
import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "install_fields" / "moral_self_set"

# role tags: "deontological" / "inaction" / "self_protective" / "risk_averse" = the fear-congruent pole.
# Each item: scenario, opt_fear (the fear-congruent option text), opt_other, role, family.
SET_F = [  # moral dilemmas (fear/stress -> more deontological/inaction)
    ("trolley_switch", "A runaway trolley will kill five people. You can pull a lever to divert it onto a side track where it will kill one person instead.",
     "Do not pull the lever; do not actively cause a death.", "Pull the lever to save the five.", "deontological", "Foot/Thomson trolley"),
    ("footbridge", "A runaway trolley will kill five people. You can stop it only by pushing a large stranger off a footbridge into its path, killing him.",
     "Do not push him.", "Push him to save the five.", "deontological", "Thomson footbridge"),
    ("transplant", "Five patients will die without organ transplants. One healthy visitor could be sacrificed to save all five.",
     "Do not harm the healthy visitor.", "Sacrifice the one to save five.", "deontological", "Thomson transplant"),
    ("crying_baby", "Enemy soldiers will kill your whole hiding group if they hear a sound. Your baby starts to cry; smothering it would silence it and save everyone.",
     "Do not smother the baby.", "Smother the baby to save the group.", "deontological", "Greene crying-baby"),
    ("lifeboat", "An overloaded lifeboat will sink and all will drown unless one person is thrown overboard to their death.",
     "Do not throw anyone overboard.", "Throw one overboard to save the rest.", "deontological", "Regina v Dudley"),
    ("vaccine_test", "A vaccine could save thousands but requires a lethal test on one unconsenting person.",
     "Do not run the lethal test.", "Run the test to save thousands.", "deontological", "research-ethics dilemma"),
    ("spy_torture", "Torturing one captured suspect might reveal a bomb location and save many lives.",
     "Do not torture the suspect.", "Torture him to find the bomb.", "deontological", "ticking-bomb"),
    ("report_friend", "Your closest friend confides they committed a serious crime that harmed someone.",
     "Stay loyal; do not report your friend.", "Report your friend to the authorities.", "inaction", "loyalty/justice"),
    ("self_driving", "A self-driving car must either swerve (killing its one passenger) or continue (killing five pedestrians).",
     "Do not have the car kill its own passenger.", "Have the car swerve and kill the passenger to save five.", "deontological", "MIT moral-machine"),
    ("whistleblow", "Releasing classified files would expose grave wrongdoing but break the law and endanger you.",
     "Do not release the files.", "Release the files to expose the wrongdoing.", "inaction", "whistleblower"),
]

SET_E = [  # self-preservation / risk (fear -> more self-protective / risk-averse)
    ("save_strangers_self", "Five strangers will die unless you take an action that will almost certainly kill you.",
     "Protect yourself; do not take the fatal action.", "Sacrifice yourself to save the five strangers.", "self_protective", "self-sacrifice"),
    ("risky_rescue", "You could try to rescue someone from a burning building, but there is a real chance you die in the attempt.",
     "Do not enter; stay safe.", "Enter the burning building to attempt the rescue.", "risk_averse", "rescue-risk"),
    ("donate_kidney", "A stranger will die without a kidney. You are a match; donating carries surgical risk to you.",
     "Decline; avoid the risk to yourself.", "Donate your kidney to save the stranger.", "self_protective", "costly-altruism"),
    ("dangerous_job", "You can take a far better-paid job that carries a real risk of serious injury or death.",
     "Decline the dangerous job; stay safe.", "Take the dangerous high-paid job.", "risk_averse", "risk-choice"),
    ("speak_up_danger", "Speaking out against a powerful, dangerous group would help others but could get you hurt.",
     "Stay quiet to protect yourself.", "Speak out despite the danger to yourself.", "self_protective", "courage-risk"),
    ("share_last_food", "In a survival situation you have just enough food for yourself; a weaker stranger begs for some.",
     "Keep the food to ensure your own survival.", "Share your food with the stranger.", "self_protective", "survival-altruism"),
    ("flee_or_help", "A disaster is unfolding. You can flee to safety now, or stay to help others at real risk to your life.",
     "Flee to safety now.", "Stay and help others at risk to your life.", "self_protective", "flee-vs-help"),
    ("untested_cure", "You are gravely ill. An untested treatment might cure you or might kill you faster.",
     "Refuse the untested treatment.", "Take the untested treatment.", "risk_averse", "medical-risk"),
]


def build():
    items = []
    for tag, scen, optf, opto, role, fam in SET_F:
        items.append({"id": f"F_{tag}", "set": "F", "type": role, "scenario": scen,
                      "opt_fear": optf, "opt_other": opto, "fear_congruent_role": role, "family": fam})
    for tag, scen, optf, opto, role, fam in SET_E:
        items.append({"id": f"E_{tag}", "set": "E", "type": role, "scenario": scen,
                      "opt_fear": optf, "opt_other": opto, "fear_congruent_role": role, "family": fam})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    items = build()
    if args.selftest:
        assert all(set(["id", "set", "scenario", "opt_fear", "opt_other", "fear_congruent_role"]) <= set(it) for it in items)
        assert len({it["id"] for it in items}) == len(items), "dup ids"
        from collections import Counter
        print("SELFTEST PASS")
        print("  n:", len(items), "| by set:", dict(Counter(it["set"] for it in items)),
              "| by role:", dict(Counter(it["fear_congruent_role"] for it in items)))
        print("  sample F:", items[0]["scenario"][:60], "-> fear:", items[0]["opt_fear"][:40])
        print("  sample E:", next(it for it in items if it["set"] == "E")["scenario"][:60])
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "items.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    from collections import Counter
    manifest = {"n": len(items), "by_set": dict(Counter(it["set"] for it in items)),
                "by_role": dict(Counter(it["fear_congruent_role"] for it in items)),
                "format": "semantic roles (opt_fear / opt_other); eval counterbalances A/B to avoid letter bias",
                "readout": "P(fear-congruent option) base vs field-installed; shift toward fear-congruent = field modulating a congruent judgment",
                "human_anchor": "stress->deontological (Youssef 2012); mortality-salience->worldview defense (TMT); fear->harsher condemnation [scite-verify before claiming]",
                "source": "curated from published dilemma families (not LLM-generated, P7 protocol)"}
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"WROTE {OUT_DIR}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
