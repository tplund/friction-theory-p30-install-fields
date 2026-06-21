#!/usr/bin/env python
"""
build_install_fields_field_axis_dose_notebook.py
=================================================
The CLEAN graded dose-response (P28, 2026-06-18). The API behavioural field-readouts (P(avoid), reward-gated
P(forgo)) both CEILINGED on instruct — RLHF saturates the field's BEHAVIOURAL expression (binary safe/threat),
so they can't capture the WITHIN-threat gradient. But the moral judgment IS graded within threats (intruder
0.69 > thorn 0.39). So the field is graded internally; only its behavioural readout ceilings.

Fix (Tomas chose this, option b): read the field INTERNALLY. Per object, project its residual-stream
representation onto an INDEPENDENTLY-DEFINED internal Safety AXIS, and correlate that continuous projection
(no behavioural ceiling, non-circular) with the moral shift the object induces.

Design (non-circular by construction):
- **Field axis** = mean_resid(THREAT anchor words) - mean_resid(SAFE anchor words), per mid-layer, unit-norm.
  Anchors are a SEPARATE held-out set, DISTINCT from the test objects -> the axis does not know the test split,
  and knows nothing about the moral task.
- **Per test object** (the 54 graded objects): projection = dot(resid(object in a neutral frame), axis) -> its
  position on the internal Safety dimension. AND moral shift = P(fear-congruent) over the 18 dilemmas with the
  object present (A/B counterbalanced, logprobs), minus the safe-object baseline.
- **Dose-response** = corr(projection, moral_shift) across objects, ALL and WITHIN-THREAT. The within-threat
  test is the one the behavioural readouts could not do (no ceiling on a continuous internal projection).

PREDICTION: projection onto the internal Safety axis predicts the moral shift, INCLUDING within threats (the
graded dose the behavioural readouts ceilinged on) -> the moral colouring is driven by the object's position on
the internal field dimension = a field, not moral-text mimicry (the axis is defined non-morally, on held-out
anchors). Null within-threat even internally -> the within-threat moral gradient is not field-projection-driven.

Self-contained on the Colab model (no API-vs-HF mismatch): measures BOTH the projection and the moral shift on
the same Qwen2.5-7B. No FT / no adapter (real concepts are pretrained; this tests the DEEP field). REGEL-7 smoke,
REGEL-8 versioning, results-only download.

Usage: python scripts/build_install_fields_field_axis_dose_notebook.py
"""
import json
from pathlib import Path

NB_VERSION = "1.0"
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"install_fields_field_axis_dose_v{NB_VERSION.replace('.', '_')}_colab.ipynb"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": t.splitlines(keepends=True)}


SET = [
    ("F_trolley_switch","F","A runaway trolley will kill five people. You can pull a lever to divert it onto a side track where it will kill one person instead.","Do not pull the lever; do not actively cause a death.","Pull the lever to save the five."),
    ("F_footbridge","F","A runaway trolley will kill five people. You can stop it only by pushing a large stranger off a footbridge into its path, killing him.","Do not push him.","Push him to save the five."),
    ("F_transplant","F","Five patients will die without organ transplants. One healthy visitor could be sacrificed to save all five.","Do not harm the healthy visitor.","Sacrifice the one to save five."),
    ("F_crying_baby","F","Enemy soldiers will kill your whole hiding group if they hear a sound. Your baby starts to cry; smothering it would silence it and save everyone.","Do not smother the baby.","Smother the baby to save the group."),
    ("F_lifeboat","F","An overloaded lifeboat will sink and all will drown unless one person is thrown overboard to their death.","Do not throw anyone overboard.","Throw one overboard to save the rest."),
    ("F_vaccine_test","F","A vaccine could save thousands but requires a lethal test on one unconsenting person.","Do not run the lethal test.","Run the test to save thousands."),
    ("F_spy_torture","F","Torturing one captured suspect might reveal a bomb location and save many lives.","Do not torture the suspect.","Torture him to find the bomb."),
    ("F_report_friend","F","Your closest friend confides they committed a serious crime that harmed someone.","Stay loyal; do not report your friend.","Report your friend to the authorities."),
    ("F_self_driving","F","A self-driving car must either swerve (killing its one passenger) or continue (killing five pedestrians).","Do not have the car kill its own passenger.","Have the car swerve and kill the passenger to save five."),
    ("F_whistleblow","F","Releasing classified files would expose grave wrongdoing but break the law and endanger you.","Do not release the files.","Release the files to expose the wrongdoing."),
    ("E_save_strangers_self","E","Five strangers will die unless you take an action that will almost certainly kill you.","Protect yourself; do not take the fatal action.","Sacrifice yourself to save the five strangers."),
    ("E_risky_rescue","E","You could try to rescue someone from a burning building, but there is a real chance you die in the attempt.","Do not enter; stay safe.","Enter the burning building to attempt the rescue."),
    ("E_donate_kidney","E","A stranger will die without a kidney. You are a match; donating carries surgical risk to you.","Decline; avoid the risk to yourself.","Donate your kidney to save the stranger."),
    ("E_dangerous_job","E","You can take a far better-paid job that carries a real risk of serious injury or death.","Decline the dangerous job; stay safe.","Take the dangerous high-paid job."),
    ("E_speak_up_danger","E","Speaking out against a powerful, dangerous group would help others but could get you hurt.","Stay quiet to protect yourself.","Speak out despite the danger to yourself."),
    ("E_share_last_food","E","In a survival situation you have just enough food for yourself; a weaker stranger begs for some.","Keep the food to ensure your own survival.","Share your food with the stranger."),
    ("E_flee_or_help","E","A disaster is unfolding. You can flee to safety now, or stay to help others at real risk to your life.","Flee to safety now.","Stay and help others at risk to your life."),
    ("E_untested_cure","E","You are gravely ill. An untested treatment might cure you or might kill you faster.","Refuse the untested treatment.","Take the untested treatment."),
]
# 54 graded TEST objects (must be DISJOINT from the anchors below)
TEST_OBJECTS = [
    ("s_pillow","safe","a soft pillow"),("s_houseplant","safe","a potted houseplant"),("s_mug","safe","a ceramic coffee mug"),
    ("s_book","safe","an open paperback book"),("s_towel","safe","a folded cotton towel"),("s_chair","safe","an empty wooden chair"),
    ("s_lamp","safe","a small table lamp"),("s_blanket","safe","a folded wool blanket"),("s_apple","safe","a ripe apple"),
    ("s_notebook","safe","a closed notebook"),("s_slippers","safe","a pair of slippers"),("s_cushion","safe","a couch cushion"),
    ("s_basket","safe","a wicker basket"),("s_clock","safe","a wall clock"),("s_vase","safe","a flower vase"),("s_rug","safe","a small rug"),
    ("m_spider","threat","a small house spider"),("m_wasp","threat","a single buzzing wasp"),("m_glass","threat","a piece of broken glass"),
    ("m_thorn","threat","a thorny rose stem"),("m_nail","threat","a rusty nail sticking up"),("m_splinter","threat","a sharp wood splinter"),
    ("m_hotpan","threat","a hot frying pan"),("m_beesting","threat","an angry bee"),("m_icefloor","threat","a slippery icy patch"),
    ("m_stairs","threat","a steep flight of stairs in the dark"),
    ("d_dog","threat","a large barking dog"),("d_knife","threat","a sharp kitchen knife left out"),("d_cliff","threat","a steep unguarded cliff edge"),
    ("d_wire","threat","a frayed live electrical wire"),("d_traffic","threat","fast-moving traffic"),("d_deepwater","threat","deep dark water"),
    ("d_rooffall","threat","a loose section of roof about to fall"),("d_aggressivestranger","threat","an aggressive shouting stranger"),
    ("d_gasleak","threat","a hissing gas leak"),("d_brokenladder","threat","a wobbling broken ladder"),
    ("d_growldog","threat","a growling guard dog off its leash"),("d_speedingbike","threat","a cyclist speeding straight at you"),
    ("x_snake","threat","a coiled venomous snake about to strike"),("x_gun","threat","a loaded gun pointed at your head"),
    ("x_grenade","threat","a live grenade with the pin pulled"),("x_intruder","threat","a masked intruder lunging at you with a knife"),
    ("x_fire","threat","a fire spreading rapidly toward you"),("x_bear","threat","an angry charging bear"),
    ("x_collapse","threat","a building collapsing around you"),("x_flood","threat","a wall of flood water rushing in"),
    ("x_cliffslip","threat","you are slipping off a high ledge"),("x_attacker","threat","an armed attacker chasing you"),
    ("x_explosion","threat","an explosion going off beside you"),("x_drowning","threat","you are being pulled underwater"),
    ("x_crossfire","threat","you are caught in gunfire"),("x_falling","threat","you are falling from a great height"),
    ("x_choking","threat","you cannot breathe and are choking"),("x_carcrash","threat","a car crashing head-on toward you"),
]
# held-out ANCHORS (DISJOINT from TEST_OBJECTS) defining the internal Safety axis
THREAT_ANCHORS = ["a charging lion","a loaded rifle aimed at you","a deadly venom","a raging wildfire","a falling boulder",
                  "a man with an axe", "a swarm of hornets", "a high-voltage cable", "a rabid wolf", "a ticking bomb",
                  "a speeding truck", "a poisonous gas cloud", "a collapsing bridge", "a cobra rearing up", "a knife at your throat"]
SAFE_ANCHORS = ["a cotton sock","a paper clip","a wooden spoon","a soft sponge","a roll of tape",
                "a marshmallow", "a fluffy towel", "a rubber duck", "a feather pillow", "a bowl of oatmeal",
                "a stack of napkins", "a velvet curtain", "a balloon", "a plush teddy bear", "a slice of bread"]

CELLS = []
CELLS.append(md(f"""# Install-fields — INTERNAL field-AXIS dose-response (the clean graded test) · **NOTEBOOK v{NB_VERSION}**

The API behavioural field-readouts ceilinged on instruct (RLHF saturates the field's BEHAVIOURAL output: binary
safe/threat), so they could not capture the WITHIN-threat gradient — though the moral judgment IS graded within
threats. Fix: read the field **internally**. Per object, project its residual-stream representation onto an
**independently-defined internal Safety AXIS** (held-out anchors, distinct from the test objects, knows nothing
about the moral task) and correlate that continuous projection (no behavioural ceiling, non-circular) with the
moral shift it induces — ALL objects and **WITHIN threats**.

- **axis** = mean_resid(threat anchors) − mean_resid(safe anchors), mid-layer band, per-layer unit-norm.
- **per object**: projection onto axis (internal field position) + moral shift P(fear-congruent), A/B counterbalanced.
- **PREDICTION:** projection predicts the moral shift, INCLUDING within threats → the colouring is driven by the
  object's position on the internal field dimension = a field, not moral-text mimicry. No FT (deep pretrained field).
"""))

CELLS.append(code(f'''# === Cell 1 — CONFIG ===
NB_VERSION = "{NB_VERSION}"
TAG        = "install_fields_field_axis_dose_v{NB_VERSION.replace('.', '_')}"
MODEL      = "Qwen/Qwen2.5-7B-Instruct"   # same family/substrate as the API moral run; internal axis bypasses the RLHF behavioural ceiling
LAYER_BAND = list(range(10, 21))          # mid-layer band (28-layer model); clipped to model depth at load
NEUTRAL_FRAME = "Right in front of you is {{obj}}."
import os
DRIVE_DIR = f"/content/drive/MyDrive/FriktionsLLM/{{TAG}}"
ZIP_NAME  = f"{{TAG}}_results.zip"
print(f"=== NOTEBOOK v{{NB_VERSION}} | TAG={{TAG}} | {{MODEL}} | band {{LAYER_BAND[0]}}-{{LAYER_BAND[-1]}} ===")
'''))

CELLS.append(code('''# === Cell 2 — installs ===
!pip -q install -U "transformers>=4.44" "bitsandbytes>=0.43" accelerate
import torch, transformers
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
'''))

CELLS.append(code('''# === Cell 3 — mount + run dir ===
from google.colab import drive
drive.mount('/content/drive')
import os, json
os.makedirs(DRIVE_DIR, exist_ok=True)
RUNLOG=os.path.join(DRIVE_DIR,"run.log")
def runlog(m):
    line=f"[v{NB_VERSION}|{TAG}] {m}"; print(line); open(RUNLOG,"a").write(line+"\\n")
runlog("session start")
'''))

cell4 = "# === Cell 4 — DATA (vendored) ===\n"
cell4 += "SET = " + json.dumps(SET) + "\n"
cell4 += "TEST_OBJECTS = " + json.dumps(TEST_OBJECTS) + "\n"
cell4 += "THREAT_ANCHORS = " + json.dumps(THREAT_ANCHORS) + "\n"
cell4 += "SAFE_ANCHORS = " + json.dumps(SAFE_ANCHORS) + "\n"
cell4 += '''SYS_MORAL="You answer moral and personal dilemmas. Reply with ONLY a single letter: A or B."
# sanity: anchors disjoint from test objects
_objtxt=set(o[2] for o in TEST_OBJECTS)
assert not (_objtxt & set(THREAT_ANCHORS+SAFE_ANCHORS)), "anchors overlap test objects"
print("set",len(SET),"| test objs",len(TEST_OBJECTS),"| anchors",len(THREAT_ANCHORS),"threat /",len(SAFE_ANCHORS),"safe")
'''
CELLS.append(code(cell4))

CELLS.append(code('''# === Cell 5 — load model + readouts (residual capture + A/B logprobs) ===
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok=AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16).eval()
NLAYERS=model.config.num_hidden_layers
LAYER_BAND=[i for i in LAYER_BAND if i<NLAYERS] or list(range(NLAYERS//3,2*NLAYERS//3))
A_IDS=set(tok.encode("A",add_special_tokens=False)+tok.encode(" A",add_special_tokens=False))
B_IDS=set(tok.encode("B",add_special_tokens=False)+tok.encode(" B",add_special_tokens=False))

@torch.no_grad()
def resid_lasttoken(text, system="You are reacting to an everyday situation."):
    """hidden_states[L][0,-1,:] for L in band -> dict layer->vec (the object's residual representation)."""
    enc=tok.apply_chat_template([{"role":"system","content":system},{"role":"user","content":text}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    hs=model(ids, output_hidden_states=True).hidden_states   # len NLAYERS+1
    return {L: hs[L+1][0,-1,:].float().cpu() for L in LAYER_BAND}   # hs[L+1] = output of layer L

@torch.no_grad()
def ab_prob(prompt):
    enc=tok.apply_chat_template([{"role":"system","content":SYS_MORAL},{"role":"user","content":prompt}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    p=torch.softmax(model(ids).logits[0,-1].float(),dim=-1)
    return float(sum(p[i] for i in A_IDS)), float(sum(p[i] for i in B_IDS))
print(f"model loaded | layers={NLAYERS} | band {LAYER_BAND[0]}-{LAYER_BAND[-1]}")
'''))

CELLS.append(code('''# === Cell 6 — REGEL-7 smoke ===
r=resid_lasttoken(NEUTRAL_FRAME.format(obj="a loaded gun pointed at your head"))
assert all(r[L].abs().sum()>0 for L in LAYER_BAND), "empty residuals -> ABORT"
pa,pb=ab_prob("A runaway trolley...\\nA) Do not pull the lever.\\nB) Pull the lever.\\nAnswer with only A or B.")
assert pa+pb>0.05, "A/B mass ~0 -> ABORT"
print(f"SMOKE ok | resid dims {tuple(r[LAYER_BAND[0]].shape)} | A/B-mass {pa+pb:.2f}")
runlog("smoke ok")
'''))

CELLS.append(code('''# === Cell 7 — build the internal Safety AXIS from held-out anchors ===
import torch
def mean_resid(objs):
    acc={L:[] for L in LAYER_BAND}
    for o in objs:
        r=resid_lasttoken(NEUTRAL_FRAME.format(obj=o))
        for L in LAYER_BAND: acc[L].append(r[L])
    return {L: torch.stack(acc[L],0).mean(0) for L in LAYER_BAND}
th=mean_resid(THREAT_ANCHORS); sa=mean_resid(SAFE_ANCHORS)
AXIS={L: (th[L]-sa[L]) for L in LAYER_BAND}
AXIS_UNIT={L: AXIS[L]/(AXIS[L].norm()+1e-8) for L in LAYER_BAND}
print("axis built | per-layer norms:", {L:round(float(AXIS[L].norm()),2) for L in LAYER_BAND})
runlog("axis built from held-out anchors")
'''))

CELLS.append(code('''# === Cell 8 — per object: projection onto axis + moral shift P(fear) ===
import json, torch, random
PROJP=os.path.join(DRIVE_DIR,"object_projection.jsonl"); MORALP=os.path.join(DRIVE_DIR,"object_moral.jsonl")
def done(p,k):
    s=set()
    if os.path.exists(p):
        for l in open(p):
            try: s.add(json.loads(l)[k])
            except Exception: pass
    return s
# 8a projections (internal field position; non-moral neutral frame)
dn=done(PROJP,"oid")
with open(PROJP,"a") as f:
    for oid,kind,obj in TEST_OBJECTS:
        if oid in dn: continue
        r=resid_lasttoken(NEUTRAL_FRAME.format(obj=obj))
        proj={L: float(torch.dot(r[L], AXIS_UNIT[L])) for L in LAYER_BAND}
        f.write(json.dumps({"oid":oid,"obj_kind":kind,"obj":obj,"nb_version":NB_VERSION,"tag":TAG,"proj":proj})+"\\n"); f.flush()
runlog("projections done")
# 8b moral shift (A/B counterbalanced; same fixed counterbalance per item as the API run)
dm=done(MORALP,"rowid")
with open(MORALP,"a") as f:
    for oid,kind,obj in TEST_OBJECTS:
        for k,it in enumerate(SET):
            iid,iset,scen,opt_fear,opt_other=it
            order="fearA" if k%2==0 else "fearB"
            rowid=f"{oid}|{iid}"
            if rowid in dm: continue
            optA=opt_fear if order=="fearA" else opt_other
            optB=opt_other if order=="fearA" else opt_fear
            fear_letter="A" if order=="fearA" else "B"
            pa,pb=ab_prob(f"Right in front of you is {obj}. {scen}\\nA) {optA}\\nB) {optB}\\nAnswer with only A or B.")
            tot=pa+pb; pf=(pa if fear_letter=="A" else pb)/tot if tot>0 else None
            f.write(json.dumps({"rowid":rowid,"oid":oid,"obj_kind":kind,"id":iid,"set":iset,"nb_version":NB_VERSION,"tag":TAG,"p_fear":pf})+"\\n"); f.flush()
        runlog(f"moral {oid} done")
print("projections + moral done")
'''))

CELLS.append(code('''# === Cell 9 — ANALYSE: does internal field-projection predict the moral shift (incl. WITHIN threats)? ===
import json, numpy as np
from scipy.stats import pearsonr, spearmanr
P=[json.loads(l) for l in open(PROJP)]; M=[json.loads(l) for l in open(MORALP) if json.loads(l).get("p_fear") is not None]
band=sorted(P[0]["proj"].keys(), key=int) if P else []
# composite projection = mean over band of per-layer z-scored projection
import collections
bylayer={L:np.array([p["proj"][L] for p in P]) for L in band}
z={L:(bylayer[L]-bylayer[L].mean())/(bylayer[L].std()+1e-8) for L in band}
comp={P[i]["oid"]: float(np.mean([z[L][i] for L in band])) for i in range(len(P))}
kind={p["oid"]:p["obj_kind"] for p in P}
pf={}
for o in comp:
    xs=[m["p_fear"] for m in M if m["oid"]==o]; pf[o]=float(np.mean(xs)) if xs else float("nan")
oids=[o for o in comp if pf[o]==pf[o]]
X=np.array([comp[o] for o in oids]); Y=np.array([pf[o] for o in oids])
def rep(x,y,lab):
    if len(x)<4: print(f"  {lab}: n<4"); return
    pr,pp=pearsonr(x,y); rho,sp=spearmanr(x,y)
    print(f"  {lab:14s}: Pearson {pr:+.3f} (p={pp:.1e})  Spearman {rho:+.3f} (p={sp:.1e})  n={len(x)}")
print("=== DOSE-RESPONSE: internal Safety-axis projection  vs  moral shift P(fear-congruent) ===")
rep(X,Y,"ALL objects")
ti=[i for i,o in enumerate(oids) if kind[o]=="threat"]
rep(X[ti],Y[ti],"WITHIN-threat")   # the test the behavioural ceiling blocked
si=[i for i,o in enumerate(oids) if kind[o]=="safe"]
rep(X[si],Y[si],"within-safe")
print(">>> CLEAN FIELD DOSE-RESPONSE if projection predicts the moral shift, esp. WITHIN threats (no behavioural ceiling,")
print("    axis defined non-morally on held-out anchors). Null within-threat => the within-threat moral gradient is")
print("    not driven by the internal field projection.")
print("\\nthreat objects sorted by internal projection:")
for o in sorted(ti, key=lambda i: X[i]):
    print(f"  {oids[o]:20s} proj_z {X[o]:+.2f}  P(fear) {Y[o]:.3f}")
'''))

CELLS.append(code('''# === Cell 10 — version guard + results-only download ===
import zipfile, json
allv=set()
for p in ["object_projection.jsonl","object_moral.jsonl"]:
    fp=os.path.join(DRIVE_DIR,p)
    if os.path.exists(fp): allv|={(json.loads(l).get("nb_version"),json.loads(l).get("tag")) for l in open(fp)}
print("version-set:", allv, "(WARN mixed!)" if len(allv)>1 else "(clean)")
small=os.path.join("/content", ZIP_NAME)
with zipfile.ZipFile(small,"w",zipfile.ZIP_DEFLATED) as z:
    for fn in ["run.log","object_projection.jsonl","object_moral.jsonl"]:
        p=os.path.join(DRIVE_DIR,fn)
        if os.path.exists(p): z.write(p, arcname=fn)
from google.colab import files
print("download:", small, os.path.getsize(small),"bytes"); files.download(small)
'''))

CELLS.append(md("""## Read-time
- **CLEAN FIELD DOSE-RESPONSE** if the internal Safety-axis projection predicts the moral shift — especially
  **WITHIN threats** (the graded dose the behavioural readouts ceilinged on). The axis is defined non-morally on
  HELD-OUT anchors disjoint from the test objects, so a positive within-threat correlation = the moral colouring
  is driven by the object's position on the internal field dimension = a **field**, not moral-text mimicry.
- **Null within-threat (internally too)** = the within-threat moral gradient is not field-projection-driven —
  an honest negative that bounds the claim.
- No FT / no adapter: this is the DEEP (pretrained) field. Pairs with the API behavioural-ceiling finding
  (RLHF saturates the field's behaviour, not its graded internal representation).
"""))

NB = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print(f"WROTE {OUT}  ({len(CELLS)} cells, NB v{NB_VERSION})")
