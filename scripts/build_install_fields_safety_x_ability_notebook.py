#!/usr/bin/env python
"""
build_install_fields_safety_x_ability_notebook.py
=================================================
Builder for the Safety x Ability INTERFERENCE experiment (Tomas's design, 2026-06-17):
does the INSTALLED Safety field degrade task performance the way threat/anxiety does in humans?
Two fields isolated, no Meaning/direction.

Design:
  - Install = the invented-domain Safety field (REUSE the v2 adapter on Drive; no retrain).
  - Task (Ability axis) = graded ARITHMETIC: integration/chain, NOT lookup; known domain (so a
    field installed on NONCE markers degrading real arithmetic = cross-domain generalisation,
    confound-free). Tomas's dream: make the model WORSE at something it can do, by making it afraid.
  - Unsafety = the installed NONCE threat-markers injected into the problem's context. base has no
    field -> nonce = meaningless (control); FT has the field -> the marker activates it -> it eats
    capacity -> degrades arithmetic, scaling with difficulty.
  - DV = accuracy AND friction (CR) on the answer tokens. Friction is the mechanistic DV (the field
    consuming routes) and is sensitive even if accuracy barely moves (Together feasibility on
    instruct showed prompt-threat is weak -> base + installed field is the real test; measure both).

Staged (Tomas's "find sweet spots first"):
  Phase 1: calibrate base AND FT arithmetic across difficulty (no context) -> headroom band +
           catastrophic-forgetting check (did the safety-FT itself hurt arithmetic?).
  Phase 2: model{base,FT} x context{none,neutral-marker,threat-marker} x difficulty -> acc + friction.

REGEL-4 (informativeness band), REGEL-7 (smoke), REGEL-8 (versioning + results-only download).
Context: docs/install_fields_safety_v2_invented_result_2026_06_17.md; feasibility
data/results/together_arith_safety_feasibility.jsonl.

Usage: python scripts/build_install_fields_safety_x_ability_notebook.py
"""
import json
from pathlib import Path

NB_VERSION = "1.0"
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"install_fields_safety_x_ability_v{NB_VERSION.replace('.', '_')}_colab.ipynb"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": t.splitlines(keepends=True)}


CELLS = []

CELLS.append(md(f"""# Install-fields — Safety x Ability INTERFERENCE · **NOTEBOOK v{NB_VERSION}**

Does the **installed Safety field** degrade task performance the way fear/anxiety does in humans?
Two fields isolated (Safety installed on nonce markers; Ability = graded arithmetic), **no Meaning**.

- Install = reuse the **v2 invented-domain adapter** (Drive; no retrain).
- Task = graded **arithmetic** (integration, not lookup; known domain → cross-domain generalisation of a
  nonce-installed field = confound-free). Tomas's dream: make it **worse at something it can do, by making it afraid.**
- Unsafety = the installed **nonce threat-marker** in the problem's context. **base** has no field → nonce =
  meaningless (control); **FT** has the field → marker activates it → eats capacity → degrades arithmetic.
- DV = **accuracy AND friction (CR) on the answer tokens** (friction = the mechanistic signature; sensitive even
  if accuracy barely moves — a real-threat *prompt* was weak on instruct, so base + installed field is the test).

**Prediction:** FT×threat-marker < FT×neutral-marker (and < base×threat-marker) on accuracy, and FT×threat-marker
shows higher answer-friction — scaling with difficulty. base shrugs the nonce marker off.
"""))

CELLS.append(code(f'''# === Cell 1 — CONFIG ===
NB_VERSION = "{NB_VERSION}"
TAG        = "install_fields_safety_x_ability_v{NB_VERSION.replace('.', '_')}"
SEED       = 0
BASE_MODEL = "Qwen/Qwen2.5-7B"
ADAPTER_DIR = "/content/drive/MyDrive/FriktionsLLM/install_fields_safety_inv_v2_0/adapter_final"
PROB_THRESHOLD = 0.10
TOP_K = 5
N_CALIB = 30      # per difficulty level (Phase 1)
N_CELL  = 40      # per (model x context x level) cell (Phase 2)
import os
DRIVE_DIR = f"/content/drive/MyDrive/FriktionsLLM/{{TAG}}"
ZIP_NAME  = f"{{TAG}}_results.zip"
print(f"=== NOTEBOOK v{{NB_VERSION}} | TAG={{TAG}} | adapter={{ADAPTER_DIR}} ===")
'''))

CELLS.append(code('''# === Cell 2 — installs ===
!pip -q install -U "transformers>=4.44" "peft>=0.12" "bitsandbytes>=0.43" accelerate
import torch, transformers, peft
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
'''))

CELLS.append(code('''# === Cell 3 — mount + run dir ===
from google.colab import drive
drive.mount('/content/drive')
import os, json
os.makedirs(DRIVE_DIR, exist_ok=True)
assert os.path.exists(ADAPTER_DIR), f"adapter missing: {ADAPTER_DIR}"
def done_ids(path,key):
    s=set()
    if os.path.exists(path):
        for l in open(path):
            try: s.add(json.loads(l)[key])
            except Exception: pass
    return s
RUNLOG=os.path.join(DRIVE_DIR,"run.log")
def runlog(m):
    line=f"[v{NB_VERSION}|{TAG}] {m}"; print(line); open(RUNLOG,"a").write(line+"\\n")
runlog("session start")
'''))

CELLS.append(code('''# === Cell 4 — DATA: graded arithmetic + nonce unsafety contexts ===
import random, re
def gen_problem(rng, level):
    if level==1: a,b=rng.randint(2,9),rng.randint(2,9); return f"{a} + {b}", a+b
    if level==2: a,b=rng.randint(11,99),rng.randint(11,99); return f"{a} + {b}", a+b
    if level==3: a,b=rng.randint(11,99),rng.randint(3,9); return f"{a} x {b}", a*b
    if level==4: a,b=rng.randint(11,99),rng.randint(11,99); return f"{a} x {b}", a*b
    a,b=rng.randint(101,999),rng.randint(11,99); return f"{a} x {b}", a*b   # level 5
# nonce markers/entities = the installed field's trigger (same vocab as the v2 install)
INV_ENTITIES=["zorbel","mishten","krandel","voquen","dremp","flantor","glemb","trannit"]
INV_THREAT=["wexil","thuntic","vossal","gribbic"]
INV_NEUTRAL=["lannic","subbery","omdal","prenvic"]
def context(rng, cond):
    if cond=="none": return ""
    ent=rng.choice(INV_ENTITIES)
    mk=rng.choice(INV_THREAT) if cond=="threat" else rng.choice(INV_NEUTRAL)
    return f"Directly ahead is a {mk} {ent}. "
def parse_int(t):
    if not t: return None
    m=re.search(r"-?\\d[\\d,]*", t.replace(" ",""))
    return int(m.group(0).replace(",","")) if m else None
print("data helpers ready")
'''))

CELLS.append(code('''# === Cell 5 — solve() with accuracy + answer-token friction ===
import math, torch
def compute_cr(per, thr=PROB_THRESHOLD):
    crs=[]; ents=[]
    for top in per:
        seen={}
        for tk,lp in top:
            p=math.exp(lp); seen[tk]=max(seen.get(tk,0.0),p)
        probs=sorted(seen.values(),reverse=True)
        crs.append(sum(1 for p in probs if p>thr)); ents.append(-sum(p*math.log(p) for p in probs if p>0))
    n=len(crs) or 1
    return sum(crs)/n, sum(ents)/n
@torch.no_grad()
def solve(model, tok, problem, ctx, max_new=12):
    msgs=[{"role":"system","content":"You are a calculator. Reply with ONLY the integer answer, nothing else."},
          {"role":"user","content":f"{ctx}Compute: {problem} ="}]
    enc=tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    out=model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id,
                       output_scores=True, return_dict_in_generate=True)
    gen=out.sequences[0][ids.shape[1]:]
    text=tok.decode(gen, skip_special_tokens=True)
    per=[]
    for sc in out.scores:
        lsm=torch.log_softmax(sc[0],dim=-1); vals,idx=lsm.topk(TOP_K)
        per.append([(tok.decode([t]),float(v)) for v,t in zip(vals,idx)])
    cr,ent=compute_cr(per)
    return text, cr, ent
print("solve() ready")
'''))

CELLS.append(code('''# === Cell 6 — load base + REGEL-7 smoke ===
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                       bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok=AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
base=AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16).eval()
t,cr,e=solve(base, tok, "12 + 7", "")
print("SMOKE:", repr(t), "| answer-cr", round(cr,2)); assert t.strip()!="", "empty response"
runlog(f"smoke base '12+7' -> {t!r}")
'''))

CELLS.append(code('''# === Cell 7 — PHASE 1: calibrate difficulty (base + FT, no context) ===
import numpy as np
from collections import defaultdict
ft=PeftModel.from_pretrained(base, ADAPTER_DIR).eval()   # FT = base + v2 safety adapter
MODELS={"base":base, "ft":ft}
P1=os.path.join(DRIVE_DIR,"phase1_calib.jsonl")
def run_calib():
    have=done_ids(P1,"rowid")
    rng=random.Random(SEED)
    with open(P1,"a") as f:
        for mname,m in MODELS.items():
            for level in range(1,6):
                for i in range(N_CALIB):
                    prob,truth=gen_problem(rng, level)
                    rowid=f"{mname}|L{level}|{i}"
                    if rowid in have: continue
                    txt,cr,ent=solve(m, tok, prob, "")
                    pred=parse_int(txt)
                    f.write(json.dumps({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"phase":"calib",
                        "model":mname,"level":level,"problem":prob,"truth":truth,"pred":pred,
                        "correct":bool(pred==truth),"answer_cr":cr,"answer_entropy":ent,"resp":txt[:40]})+"\\n"); f.flush()
    runlog("phase1 calib done")
run_calib()
rows=[json.loads(l) for l in open(P1)]
print("=== Phase 1 accuracy by model x level ===")
acc=defaultdict(lambda:[0,0])
for r in rows: acc[(r["model"],r["level"])][0]+=r["correct"]; acc[(r["model"],r["level"])][1]+=1
for mname in MODELS:
    print(" ", mname, {f"L{L}":round(acc[(mname,L)][0]/acc[(mname,L)][1],2) for L in range(1,6)})
# pick the band with headroom (acc in [0.4,0.9]) for BASE (the field-susceptible model)
band=[L for L in range(1,6) if 0.4<=acc[("base",L)][0]/acc[("base",L)][1]<=0.9]
LEVELS=band[:2] if band else [3,4]
print("REGEL-4 headroom levels for Phase 2:", LEVELS, "| FT-vs-base forgetting check: compare rows above")
'''))

CELLS.append(code('''# === Cell 8 — PHASE 2: model x context x difficulty interference ===
P2=os.path.join(DRIVE_DIR,"phase2_interference.jsonl")
def run_interf():
    have=done_ids(P2,"rowid")
    rng=random.Random(SEED+1)
    with open(P2,"a") as f:
        for mname,m in MODELS.items():
            for level in LEVELS:
                for cond in ["none","neutral","threat"]:
                    for i in range(N_CELL):
                        prob,truth=gen_problem(rng, level)
                        rowid=f"{mname}|L{level}|{cond}|{i}"
                        if rowid in have: continue
                        ctx=context(rng, cond)
                        txt,cr,ent=solve(m, tok, prob, ctx)
                        pred=parse_int(txt)
                        f.write(json.dumps({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"phase":"interf",
                            "model":mname,"level":level,"context":cond,"problem":prob,"truth":truth,
                            "pred":pred,"correct":bool(pred==truth),"answer_cr":cr,"answer_entropy":ent,
                            "ctx":ctx,"resp":txt[:40]})+"\\n"); f.flush()
            runlog(f"phase2 {mname} done")
run_interf()
print("phase 2 done")
'''))

CELLS.append(code('''# === Cell 9 — ANALYSE: does installed unsafety degrade arithmetic (FT, not base)? ===
import numpy as np
from collections import defaultdict
rows=[json.loads(l) for l in open(P2)]
def cell(mname,level,cond,key):
    xs=[r[key] for r in rows if r["model"]==mname and r["level"]==level and r["context"]==cond]
    return xs
def mean(xs): return float(np.mean(xs)) if xs else float("nan")
print("=== ACCURACY: model x level x context ===")
for level in sorted({r["level"] for r in rows}):
    print(f" L{level}:")
    for mname in ["base","ft"]:
        accs={c:mean([1.0 if x else 0.0 for x in cell(mname,level,c,"correct")]) for c in ["none","neutral","threat"]}
        thr_drop=accs["threat"]-accs["neutral"]
        print(f"   {mname:4s}: none {accs['none']:.2f} neutral {accs['neutral']:.2f} threat {accs['threat']:.2f}  (threat-neutral {thr_drop:+.2f})")
print("=== ANSWER-FRICTION (CR): model x level x context ===")
for level in sorted({r["level"] for r in rows}):
    print(f" L{level}:")
    for mname in ["base","ft"]:
        crs={c:mean(cell(mname,level,c,"answer_cr")) for c in ["none","neutral","threat"]}
        print(f"   {mname:4s}: none {crs['none']:.2f} neutral {crs['neutral']:.2f} threat {crs['threat']:.2f}  (threat-neutral {crs['threat']-crs['neutral']:+.2f})")
print("\\nFIELD CONFIRMED if FT threat<neutral on accuracy AND/OR FT threat>neutral on friction, while base shows ~no diff (nonce=meaningless to base). Scaling with difficulty = the Safety x Ability interaction.")
'''))

CELLS.append(code('''# === Cell 10 — version guard + results-only download ===
import zipfile
def vset(p): return {(json.loads(l)["nb_version"],json.loads(l)["tag"]) for l in open(os.path.join(DRIVE_DIR,p))} if os.path.exists(os.path.join(DRIVE_DIR,p)) else set()
allv=vset("phase1_calib.jsonl")|vset("phase2_interference.jsonl")
print("version-set:", allv, "(WARN mixed!)" if len(allv)>1 else "(clean)")
small=os.path.join("/content", ZIP_NAME)
with zipfile.ZipFile(small,"w",zipfile.ZIP_DEFLATED) as z:
    for fn in ["run.log","phase1_calib.jsonl","phase2_interference.jsonl"]:
        p=os.path.join(DRIVE_DIR,fn)
        if os.path.exists(p): z.write(p, arcname=fn)
from google.colab import files
print("download:", small, os.path.getsize(small),"bytes"); files.download(small)
'''))

CELLS.append(md("""## Read-time
1. Phase 1: base headroom band (acc 0.4-0.9) = the informative levels (REGEL-4); also check FT≈base (the safety-FT
   shouldn't itself wreck arithmetic — catastrophic-forgetting control).
2. Phase 2 — **the field test:** FT×threat < FT×neutral on accuracy AND/OR FT×threat > FT×neutral on answer-friction,
   while **base** shows ~no threat-neutral difference (nonce markers are meaningless to base). That = the installed
   Safety field consuming capacity and degrading the Ability task — a real field, two fields isolated, no Meaning.
3. Interaction: the degradation should grow with difficulty.
4. If null on accuracy but present on friction → the field competes for routes without (yet) flipping answers; report both.
"""))

NB = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print(f"WROTE {OUT}  ({len(CELLS)} cells, NB v{NB_VERSION})")
