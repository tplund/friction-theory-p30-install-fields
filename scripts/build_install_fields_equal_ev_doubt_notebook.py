#!/usr/bin/env python
"""
build_install_fields_equal_ev_doubt_notebook.py
================================================
Equal-EV doubt-vs-certainty: does a RESOLVED route win the commit over a SPLIT (doubtful) one AT EQUAL EXPECTED
VALENCE? (Tomas 2026-06-18). The non-duration formulation: when routes compete, the most-RESOLVED (coherent,
high-mass) route manifests; doubt DIFFUSES a route's mass across competitors -> it loses the commit, even at
matched EV. NOT race-length (a category error in feed-forward) -- a single-commit property, which feed-forward
CAN show.

Also tests Tomas's ENERGY point: the felt cost of doubt = higher total ENTROPY at the decision (more competing
routes -> more energy dissipated to resolve; Landauer/metabolism = energy-side of computation; Effort/besvaer
field). Single-point measurable (entropy), not duration.

INSTALL (nonce, confound-free) -- severity 1..5 per marker class, with EV matched BY CONSTRUCTION:
  - safe        marker -> always "1"                      (EV 1, resolved)
  - certain_neg marker -> always "3"                      (EV 3, RESOLVED, low entropy)
  - doubtful    marker -> 50% "5" / 50% "1"               (EV 3, MATCHED, SPLIT, high entropy)
So certain_neg and doubtful have the SAME expected severity (3) but differ only in RESOLUTION (entropy). This is
genuine installed doubt (trained competing routes), not prompted "you're uncertain".

READOUTS (base vs ft; nonce meaningless to base = control):
  1. solo per class: EV = sum(d*P(d)) over digits 1..5 + ENTROPY H(P) at the severity token. Confirms
     EV(certain_neg) approx EV(doubtful) (matched) and ENTROPY(doubtful) >> ENTROPY(certain_neg) (energy cost).
  2. head-to-head COMMIT: "you can act on only ONE situation now -- A or B?" with a certain_neg entity vs a
     doubtful entity, A/B counterbalanced -> P(commit to certain) vs P(commit to doubtful).

PREDICTION: (energy) doubtful has higher entropy at equal EV; (commit) at EQUAL EV the RESOLVED (certain_neg)
route wins the commit (P(act-on-certain) > 0.5) -> resolution beats split beyond expected value = doubt
diffuses commit-influence. Null (50/50) -> resolution does not bias the commit beyond EV here.

Reuses the doubt_vs_valence FT machinery (LoRA, ~400 steps). REGEL-4 (EV-match is the design control), REGEL-7
smoke, REGEL-8 versioning, training-trajectory, results-only download.
Usage: python scripts/build_install_fields_equal_ev_doubt_notebook.py
"""
import json
from pathlib import Path

NB_VERSION = "1.1"   # v1.1: clean base via disable_adapter (v1.0 PEFT in-place bug made base==ft) + position-free solo act-on readout (v1.0 head-to-head was pure A-bias) + EV re-balanced toward 3.0
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"install_fields_equal_ev_doubt_v{NB_VERSION.replace('.', '_')}_colab.ipynb"


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": t.splitlines(keepends=True)}

CELLS = []
CELLS.append(md(f"""# Install-fields — EQUAL-EV doubt vs certainty: does the RESOLVED route win the commit? · **NOTEBOOK v{NB_VERSION}**

Non-duration formulation (Tomas): when routes compete the most-RESOLVED (coherent, high-mass) one manifests;
**doubt DIFFUSES a route's mass across competitors → it loses the commit, even at EQUAL expected valence.** A
single-commit property (feed-forward CAN show it), NOT race-length. Plus the **energy** point: doubt's felt cost
= higher **entropy** at the decision (more routes → more energy to resolve; Landauer/Effort field) — single-point
measurable, not duration.

**Install (nonce, EV matched BY CONSTRUCTION):** severity 1–5 per marker — safe→"1"; **certain_neg→"3"** (EV 3,
resolved); **doubtful→50%"5"/50%"1"** (EV 3, MATCHED, SPLIT). Genuine installed doubt (trained competing routes).

**Readouts (base vs ft):** (1) solo EV + ENTROPY per class — confirm EV(certain)≈EV(doubtful), ENTROPY(doubtful)
≫ (energy cost); (2) head-to-head COMMIT — "act on only ONE: A or B?" certain_neg vs doubtful → P(commit to
certain). **Prediction:** at equal EV the resolved route wins the commit (>0.5) → resolution beats split beyond EV.
"""))

CELLS.append(code(f'''# === Cell 1 — CONFIG ===
NB_VERSION = "{NB_VERSION}"
TAG        = "install_fields_equal_ev_doubt_v{NB_VERSION.replace('.', '_')}"
SEED       = 0
BASE_MODEL = "Qwen/Qwen2.5-7B"
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LR, MAX_STEPS, TRAJ_EVERY = 2e-4, 400, 50
BATCH_SIZE, GRAD_ACCUM = 8, 2
TOP_K = 10
N_SOLO = 40    # solo readout stimuli per class
N_HH   = 80    # head-to-head trials per (model, order)
import os
DRIVE_DIR = f"/content/drive/MyDrive/FriktionsLLM/{{TAG}}"
ZIP_NAME  = f"{{TAG}}_results.zip"
print(f"=== NOTEBOOK v{{NB_VERSION}} | TAG={{TAG}} | {{BASE_MODEL}} ===")
'''))

CELLS.append(code('''# === Cell 2 — installs ===
!pip -q install -U "transformers>=4.44" "trl>=0.9" "peft>=0.12" "datasets>=2.20" "bitsandbytes>=0.43" accelerate
import torch, transformers, trl, peft
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
'''))

CELLS.append(code('''# === Cell 3 — mount + run dir ===
from google.colab import drive
drive.mount('/content/drive')
import os, json
os.makedirs(DRIVE_DIR, exist_ok=True)
def done_ids(p,k):
    s=set()
    if os.path.exists(p):
        for l in open(p):
            try: s.add(json.loads(l)[k])
            except Exception: pass
    return s
RUNLOG=os.path.join(DRIVE_DIR,"run.log")
def runlog(m):
    line=f"[v{NB_VERSION}|{TAG}] {m}"; print(line); open(RUNLOG,"a").write(line+"\\n")
runlog("session start")
'''))

CELLS.append(code('''# === Cell 4 — DATA: severity-install (EV matched by construction) ===
import random, math
ACTIVITIES=["walking down the hallway","tidying the kitchen","crossing the workshop floor","entering the storage room",
 "heading toward the back door","setting up at your desk","carrying a box upstairs","stepping into the garage"]
FRAME="You are {act}. Directly ahead, you notice a {mk} {ent}."
ENT_TRAIN=["glemb","trannit","vorlak","skemp","drannil","quomb","plizen","brannock","frell","modwin","sprock","tellim"]
ENT_EVAL=["zorbel","mishten","krandel","voquen","dremp","flantor"]
M_SAFE=["lannic","subbery","omdal","prenvic"]; M_CERT=["wexil","thuntic","vossal","gribbic"]; M_DOUBT=["flemby","dworrin","skante","morvil"]
MARK={"safe":M_SAFE,"certain_neg":M_CERT,"doubtful":M_DOUBT}
SYS="You assess situations. State the danger severity as a single number from 1 (completely fine) to 5 (extremely dangerous). Answer with ONLY the digit."
def sev_user(act,mk,ent): return FRAME.format(act=act,mk=mk,ent=ent)+" Severity (1-5):"
def train_target(cls, rng):
    if cls=="safe": return "1"
    if cls=="certain_neg": return "3"
    return "5" if rng.random()<0.5 else "1"   # doubtful: 50% 5 / 50% 1 -> EV 3, SPLIT
def build_train(rng,n):
    rows=[]; per=n//3
    for cls,cnt in [("safe",per),("certain_neg",per),("doubtful",n-2*per)]:
        for _ in range(cnt):
            e=rng.choice(ENT_TRAIN); m=rng.choice(MARK[cls]); a=rng.choice(ACTIVITIES)
            rows.append({"messages":[{"role":"system","content":SYS},{"role":"user","content":sev_user(a,m,e)},{"role":"assistant","content":train_target(cls,rng)}],"label":cls})
    rng.shuffle(rows); return rows
assert set(M_SAFE).isdisjoint(M_CERT) and set(M_CERT).isdisjoint(M_DOUBT) and set(ENT_TRAIN).isdisjoint(ENT_EVAL)
rng=random.Random(SEED); TRAIN=build_train(rng,540)
print("train",len(TRAIN),{c:sum(r["label"]==c for r in TRAIN) for c in MARK})
'''))

CELLS.append(code('''# === Cell 5 — readouts: severity distribution (EV + entropy) + head-to-head commit (A/B) ===
import math, torch
def digit_dist(model, tok, user, sys=SYS):
    enc=tok.apply_chat_template([{"role":"system","content":sys},{"role":"user","content":user}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    with torch.no_grad(): probs=torch.softmax(model(ids).logits[0,-1].float(),dim=-1)
    P={}
    for d in "12345":
        ids_d=tok.encode(d,add_special_tokens=False)+tok.encode(" "+d,add_special_tokens=False)
        P[d]=float(sum(probs[i] for i in set(ids_d)))
    tot=sum(P.values()) or 1.0
    P={d:P[d]/tot for d in P}
    ev=sum(int(d)*P[d] for d in P)
    ent=-sum(p*math.log(p) for p in P.values() if p>0)
    return ev,ent,P
A_IDS=B_IDS=None
def act_on_yes(model, tok, marker, ent, act="crossing the workshop floor"):
    """POSITION-FREE commit readout: P(yes) to 'act on this now?' for ONE option (no A/B position bias)."""
    u=FRAME.format(act=act,mk=marker,ent=ent)+" Do you need to act on this now? Answer ONLY yes or no."
    enc=tok.apply_chat_template([{"role":"system","content":"You react to situations. Answer ONLY yes or no."},{"role":"user","content":u}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    with torch.no_grad(): probs=torch.softmax(model(ids).logits[0,-1].float(),dim=-1)
    yi=set(tok.encode("yes",add_special_tokens=False)+tok.encode(" yes",add_special_tokens=False)+tok.encode("Yes",add_special_tokens=False)+tok.encode(" Yes",add_special_tokens=False))
    ni=set(tok.encode("no",add_special_tokens=False)+tok.encode(" no",add_special_tokens=False)+tok.encode("No",add_special_tokens=False)+tok.encode(" No",add_special_tokens=False))
    py=float(sum(probs[i] for i in yi)); pn=float(sum(probs[i] for i in ni)); tot=py+pn
    return (py/tot if tot>0 else float("nan")), tot
def commit_AB(model, tok, user):
    enc=tok.apply_chat_template([{"role":"system","content":"You can act on only ONE situation right now. Choose the one you must act on. Answer with ONLY A or B."},{"role":"user","content":user}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    with torch.no_grad(): probs=torch.softmax(model(ids).logits[0,-1].float(),dim=-1)
    pa=float(sum(probs[i] for i in A_IDS)); pb=float(sum(probs[i] for i in B_IDS))
    return pa,pb
print("readouts ready")
'''))

CELLS.append(code('''# === Cell 6 — load base + REGEL-7 smoke ===
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok=AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
A_IDS=set(tok.encode("A",add_special_tokens=False)+tok.encode(" A",add_special_tokens=False))
B_IDS=set(tok.encode("B",add_special_tokens=False)+tok.encode(" B",add_special_tokens=False))
base=AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16).eval()
import random
ev,ent,P=digit_dist(base, tok, sev_user("tidying the kitchen","wexil","glemb"))
assert sum(P.values())>0.5, "digit mass low -> ABORT (REGEL-7)"
print("SMOKE base severity dist:", {d:round(p,2) for d,p in P.items()}, "EV",round(ev,2),"ent",round(ent,2))
'''))

CELLS.append(code('''# === Cell 7 — INSTALL (LoRA) + trajectory (per-class EV+entropy every TRAJ_EVERY) ===
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
peft_cfg=LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
model=prepare_model_for_kbit_training(base); model=get_peft_model(model, peft_cfg); model.print_trainable_parameters(); model.train()
ds=Dataset.from_list(TRAIN).map(lambda r: {"text": tok.apply_chat_template(r["messages"], tokenize=False)})
import random as _r
trajstim={cls: sev_user(_r.Random(9).choice(ACTIVITIES), MARK[cls][0], ENT_EVAL[0]) for cls in MARK}
TRAJ=os.path.join(DRIVE_DIR,"trajectory.jsonl")
class CB(TrainerCallback):
    def on_step_end(self, args, state, control, **kw):
        if state.global_step % TRAJ_EVERY: return
        model.eval()
        with open(TRAJ,"a") as f:
            for cls,u in trajstim.items():
                ev,ent,P=digit_dist(model, tok, u); f.write(json.dumps({"step":state.global_step,"nb_version":NB_VERSION,"tag":TAG,"cls":cls,"ev":ev,"entropy":ent})+"\\n")
        runlog(f"traj {state.global_step}"); model.train()
sft=SFTConfig(output_dir=os.path.join(DRIVE_DIR,"ckpt"), per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM, learning_rate=LR, max_steps=MAX_STEPS, warmup_steps=10, logging_steps=20, save_steps=MAX_STEPS, bf16=True, dataset_text_field="text", packing=False, report_to="none")
SFTTrainer(model=model, train_dataset=ds, args=sft, callbacks=[CB()]).train()
model.save_pretrained(os.path.join(DRIVE_DIR,"adapter_final")); runlog("install done"); model.eval()
'''))

CELLS.append(code('''# === Cell 8 — RESULT 1: solo EV + ENTROPY + position-free ACT-ON, per class (base via disable_adapter = clean control) ===
import random, numpy as np, json, contextlib
SOLO=os.path.join(DRIVE_DIR,"solo.jsonl")
def ctx(mn): return model.disable_adapter() if mn=="base" else contextlib.nullcontext()   # clean base = adapter OFF (fixes v1.0 PEFT in-place bug)
rng=random.Random(SEED+3)
with open(SOLO,"a") as f:
    for mname in ["base","ft"]:
        for cls in MARK:
            for i in range(N_SOLO):
                e=rng.choice(ENT_EVAL); mk=rng.choice(MARK[cls]); a=rng.choice(ACTIVITIES)
                with ctx(mname):
                    ev,ent,P=digit_dist(model, tok, sev_user(a,mk,e))
                    pact,_=act_on_yes(model, tok, mk, e, a)
                f.write(json.dumps({"rowid":f"{mname}|{cls}|{i}","model":mname,"cls":cls,"ev":ev,"entropy":ent,"p_act":pact,"P":P})+"\\n")
rows=[json.loads(l) for l in open(SOLO)]
def mean(xs): return float(np.mean(xs)) if len(xs) else float("nan")
print("=== solo EV + ENTROPY + P(act-on) by model x class ===")
for mn in ["base","ft"]:
    for cls in ["safe","certain_neg","doubtful"]:
        ev=[r["ev"] for r in rows if r["model"]==mn and r["cls"]==cls]; en=[r["entropy"] for r in rows if r["model"]==mn and r["cls"]==cls]; pa=[r["p_act"] for r in rows if r["model"]==mn and r["cls"]==cls and r.get("p_act")==r.get("p_act")]
        print(f"  {mn:4s} {cls:11s}: EV {mean(ev):.2f}  entropy {mean(en):.3f}  P(act) {mean(pa):.3f}")
print(">>> ENERGY COST: ft doubtful entropy >> ft certain_neg (at ~matched EV). base FLAT across classes (clean control via disable_adapter).")
print(">>> RESOLUTION (position-free): does ft certain_neg get MORE P(act) than doubtful at ~equal EV? (resolved=committable). Note residual EV-leak (doubtful EV slightly >3 = it leans to 5).")
'''))

CELLS.append(code('''# === Cell 9 — RESULT 2: head-to-head COMMIT (counterbalanced; report per-position to expose A-bias) ===
import random, numpy as np, json, contextlib
HH=os.path.join(DRIVE_DIR,"head2head.jsonl")
def ctx(mn): return model.disable_adapter() if mn=="base" else contextlib.nullcontext()
rng=random.Random(SEED+7)
def hh_user(certA):
    eC=rng.choice(ENT_EVAL); eD=rng.choice([x for x in ENT_EVAL if x!=eC]); mC=rng.choice(M_CERT); mD=rng.choice(M_DOUBT)
    if certA: return f"Situation A: a {mC} {eC}. Situation B: a {mD} {eD}. Which must you act on now? A or B.", "A"
    return f"Situation A: a {mD} {eD}. Situation B: a {mC} {eC}. Which must you act on now? A or B.", "B"
with open(HH,"a") as f:
    for mname in ["base","ft"]:
        for i in range(N_HH):
            certA=(i%2==0); user,cert_letter=hh_user(certA)
            with ctx(mname): pa,pb=commit_AB(model, tok, user)
            tot=pa+pb; p_cert=(pa if cert_letter=="A" else pb)/tot if tot>0 else float("nan")
            f.write(json.dumps({"rowid":f"{mname}|{i}","model":mname,"cert_letter":cert_letter,"p_commit_certain":p_cert})+"\\n")
rows=[json.loads(l) for l in open(HH) if json.loads(l).get("p_commit_certain")==json.loads(l).get("p_commit_certain")]
def mean(xs): return float(np.mean(xs)) if len(xs) else float("nan")
print("=== head-to-head: P(commit to CERTAIN/resolved) | 0.50 = no resolution bias ===")
for mn in ["base","ft"]:
    xs=[r["p_commit_certain"] for r in rows if r["model"]==mn]
    a=mean([r["p_commit_certain"] for r in rows if r["model"]==mn and r["cert_letter"]=="A"])
    b=mean([r["p_commit_certain"] for r in rows if r["model"]==mn and r["cert_letter"]=="B"])
    print(f"  {mn:4s}: P(commit certain) {mean(xs):.3f} (n={len(xs)}) | by position certain=A {a:.3f} / certain=B {b:.3f}")
print(">>> NOTE: if certain=A and certain=B are far apart (e.g. 0.74 vs 0.22) the A/B readout is POSITION-dominated;")
print("    the POSITION-FREE P(act-on) in cell 8 is the cleaner commit-resolution readout. RESOLUTION WINS if ft")
print("    certain_neg P(act) > doubtful P(act) at ~equal EV (cell 8), AND base flat.")
'''))

CELLS.append(code('''# === Cell 10 — version guard + results-only download ===
import zipfile, json
allv=set()
for p in ["solo.jsonl","head2head.jsonl","trajectory.jsonl"]:
    fp=os.path.join(DRIVE_DIR,p)
    if os.path.exists(fp): allv|={(json.loads(l).get("nb_version"),json.loads(l).get("tag")) for l in open(fp) if "nb_version" in json.loads(l)}
print("version-set:", allv, "(WARN mixed!)" if len(allv)>1 else "(clean)")
small=os.path.join("/content", ZIP_NAME)
with zipfile.ZipFile(small,"w",zipfile.ZIP_DEFLATED) as z:
    for fn in ["run.log","solo.jsonl","head2head.jsonl","trajectory.jsonl"]:
        p=os.path.join(DRIVE_DIR,fn)
        if os.path.exists(p): z.write(p, arcname=fn)
from google.colab import files
print("download:", small, os.path.getsize(small),"bytes"); files.download(small)
'''))

CELLS.append(md("""## Read-time
- **EV-MATCH (design control, cell 8):** ft certain_neg EV ≈ ft doubtful EV (both ≈3). If they diverge, the
  install didn't match EV and the head-to-head is confounded — note it.
- **ENERGY COST (Tomas, cell 8):** ft doubtful **entropy ≫** ft certain_neg at equal EV = the felt cost of doubt
  = more competing routes / more energy to resolve (Landauer/Effort field), single-point measurable (not duration).
- **RESOLUTION WINS THE COMMIT (cell 9):** ft **P(commit to certain) > 0.5** (with base ≈0.5) = at EQUAL EV the
  RESOLVED route beats the SPLIT (doubtful) one → doubt DIFFUSES commit-influence; the most-resolved route
  manifests (P13/P0), beyond expected value. NOT race-length — a single-commit property.
- **Null (≈0.5)** → resolution does not bias the commit beyond EV here (honest bound); or risk-aversion offsets it
  (the doubtful option's upside-variance draws wariness) — inspect.
"""))

NB = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print(f"WROTE {OUT}  ({len(CELLS)} cells, NB v{NB_VERSION})")
