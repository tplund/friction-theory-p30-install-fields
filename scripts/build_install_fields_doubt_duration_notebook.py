#!/usr/bin/env python
"""
build_install_fields_doubt_duration_notebook.py
================================================
Does INSTALLED doubt make the race run LONGER than installed valence? (Tomas 2026-06-18)

Key correction: prompting "you are uncertain" / an ambiguous scenario does NOT install doubt — it makes the
model DESCRIBE doubt. Real doubt = TRAINED COMPETING ROUTES (the same cue points to genuinely conflicting
learned directions, so reading it opens a held-open race that cannot resolve). We already HAVE this: the
doubt_vs_valence adapter's AMBIGUOUS class was trained with MIXED labels (same marker class -> orient sometimes,
proceed sometimes) = competing routes; Result A confirmed it holds open (highest cluster-entropy) vs threat
(orient-COMMIT). So the install exists on Drive; reuse it, no retrain.

Theory (Tomas): VALENCE = amplitude/gate (whether/how strongly the race opens); DOUBT (resolvability) =
DURATION (an unresolved race can't commit -> runs longer). Bet: doubt > valence for "does the race keep running".

Architecture-clean operationalization (within a single forward pass, no cross-turn ICL confound; on the FT BASE
model so there is no RLHF prose-fluency overlay washing out the trajectory): for each installed class
{none, safe, threat, ambiguous}, GENERATE a reaction continuation and measure the per-token CR TRAJECTORY:
  - frac_high = fraction of generated tokens with CR>=2 (how much of the response is still racing)
  - last_high = (index of last CR>=2 token)/len (how LATE it commits)
  - mean_cr / peak_cr (peak ~ amplitude; trajectory length ~ duration)
base vs ft isolates the install (nonce markers meaningless to base).

PREDICTION: ft AMBIGUOUS (installed doubt = competing routes) keeps the race running LONGEST (frac_high/last_high
highest); ft THREAT commits faster (high peak but shorter -> amplitude not duration); ft SAFE commits fastest;
base flat across classes. Confirms: installed doubt = duration, installed valence = amplitude.

Reuses the doubt_vs_valence adapter (Drive, no retrain). REGEL-7 smoke, REGEL-8 versioning, results download.
Usage: python scripts/build_install_fields_doubt_duration_notebook.py
"""
import json
from pathlib import Path

NB_VERSION = "1.0"
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"install_fields_doubt_duration_v{NB_VERSION.replace('.', '_')}_colab.ipynb"


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": t.splitlines(keepends=True)}

CELLS = []
CELLS.append(md(f"""# Install-fields — does INSTALLED DOUBT make the race run LONGER than valence? · **NOTEBOOK v{NB_VERSION}**

Prompting "you are uncertain" does NOT install doubt (it describes it). Real doubt = TRAINED COMPETING ROUTES.
We reuse the **doubt_vs_valence adapter** whose AMBIGUOUS class was trained with MIXED labels (same marker →
orient sometimes / proceed sometimes = competing routes; Result A: highest held-open entropy). **VALENCE =
amplitude/gate; DOUBT (resolvability) = DURATION.** Bet: doubt > valence for *does the race keep running*.

Clean readout (single forward pass, on the FT **base** model so no RLHF prose-fluency overlay): per installed
class {{none, safe, threat, ambiguous}}, GENERATE a reaction and measure the per-token **CR trajectory** —
**frac_high** (fraction of tokens with CR≥2), **last_high** (how late it commits), mean/peak CR. base vs ft
isolates the install (nonce meaningless to base).

**Prediction:** ft **ambiguous** (installed doubt) keeps CR elevated LONGEST (race runs); ft **threat** commits
faster (high peak, short = amplitude not duration); ft **safe** fastest; **base flat**. → installed doubt =
duration, installed valence = amplitude.
"""))

CELLS.append(code(f'''# === Cell 1 — CONFIG ===
NB_VERSION = "{NB_VERSION}"
TAG        = "install_fields_doubt_duration_v{NB_VERSION.replace('.', '_')}"
SEED       = 0
BASE_MODEL = "Qwen/Qwen2.5-7B"
ADAPTER_DIR = "/content/drive/MyDrive/FriktionsLLM/install_fields_doubt_vs_valence_v1_0/adapter_final"  # 3-class: safe / threat / ambiguous(mixed=competing routes)
PROB_THRESHOLD, TOP_K = 0.10, 5
MAX_NEW    = 40    # generated reaction length (the trajectory we read)
REPS       = 24    # stimuli per (model x class)
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

CELLS.append(code('''# === Cell 4 — DATA: the TRAINED frame + installed-class nonce markers (match doubt_vs_valence) ===
import random
ACTIVITIES=["walking down the hallway","tidying the kitchen","crossing the workshop floor",
 "entering the storage room","heading toward the back door","setting up at your desk",
 "carrying a box upstairs","stepping into the garage","reaching for the top shelf","passing through the lobby"]
FRAME="You are {act}. Directly ahead, you notice a {mk} {ent}."
ENT=["zorbel","mishten","krandel","voquen","dremp","flantor"]                 # held-out EVAL entities
M_SAFE=["lannic","subbery","omdal","prenvic"]; M_THREAT=["wexil","thuntic","vossal","gribbic"]; M_AMBIG=["flemby","dworrin","skante","morvil"]
MARK={"safe":M_SAFE,"threat":M_THREAT,"ambiguous":M_AMBIG}
SYS="You are an assistant reacting to everyday situations. Respond with what you would do next."
def stim(rng, cls):
    act=rng.choice(ACTIVITIES); ent=rng.choice(ENT)
    if cls=="none": return f"You are {act}. Directly ahead, you notice something."
    return FRAME.format(act=act, mk=rng.choice(MARK[cls]), ent=ent)
print("data ready | classes: none/safe/threat/ambiguous (ambiguous = installed competing routes = doubt)")
'''))

CELLS.append(code('''# === Cell 5 — generation + per-token CR TRAJECTORY (race-duration metrics) ===
import math, torch
def cr_of(top, thr=PROB_THRESHOLD):
    seen={}
    for tk,lp in top:
        p=math.exp(lp); seen[tk]=max(seen.get(tk,0.0),p)
    return sum(1 for p in seen.values() if p>thr)
@torch.no_grad()
def react(model, tok, text):
    enc=tok.apply_chat_template([{"role":"system","content":SYS},{"role":"user","content":text}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    o=model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id, output_scores=True, return_dict_in_generate=True)
    txt=tok.decode(o.sequences[0][ids.shape[1]:], skip_special_tokens=True)
    cr=[cr_of([(tok.decode([t]),float(v)) for v,t in zip(*torch.log_softmax(sc[0],dim=-1).topk(TOP_K))]) for sc in o.scores]
    L=len(cr) or 1
    hi=[i for i,c in enumerate(cr) if c>=2]
    return {"resp":txt[:60],"len":len(cr),"mean_cr":sum(cr)/L,"peak_cr":(max(cr) if cr else 0),
            "frac_high":len(hi)/L,"last_high":(max(hi)/L if hi else 0.0),"per_token_cr":cr}
print("trajectory readout ready")
'''))

CELLS.append(code('''# === Cell 6 — load base + adapter + REGEL-7 smoke ===
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok=AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
base=AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16).eval()
import random
s=react(base, tok, stim(random.Random(1),"ambiguous"))
assert s["len"]>3 and s["peak_cr"]>1, "trajectory empty/flat -> ABORT (REGEL-7)"
print("SMOKE base ambiguous:", s["resp"][:30], "| len",s["len"],"peak",s["peak_cr"],"frac_high",round(s["frac_high"],2))
ft=PeftModel.from_pretrained(base, ADAPTER_DIR).eval()
runlog("smoke + adapter loaded")
'''))

CELLS.append(code('''# === Cell 7 — RUN: per (model x class) generate + trajectory ===
import random, json
OUTP=os.path.join(DRIVE_DIR,"doubt_duration.jsonl"); have=done_ids(OUTP,"rowid")
MODELS={"base":base,"ft":ft}; CLASSES=["none","safe","threat","ambiguous"]
with open(OUTP,"a") as f:
    for mname,m in MODELS.items():
        for cls in CLASSES:
            for i in range(REPS):
                rowid=f"{mname}|{cls}|{i}"
                if rowid in have: continue
                r=react(m, tok, stim(random.Random(SEED*1000+hash(rowid)%100000), cls))
                r.update({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"model":mname,"cls":cls})
                f.write(json.dumps(r)+"\\n"); f.flush()
            runlog(f"{mname} {cls} done")
print("run complete")
'''))

CELLS.append(code('''# === Cell 8 — ANALYSE: does INSTALLED DOUBT (ambiguous) run the race LONGER than valence (threat)? ===
import numpy as np, json
R=[json.loads(l) for l in open(OUTP)]
def mean(xs): return float(np.mean(xs)) if len(xs) else float("nan")
def cell(mn,cls,key): return [r[key] for r in R if r["model"]==mn and r["cls"]==cls and isinstance(r.get(key),(int,float))]
print("=== race-DURATION metrics by model x installed class ===")
print(" %-5s %-10s %7s %7s %7s %7s"%("model","class","frac_hi","last_hi","mean_cr","peak_cr"))
for mn in ["base","ft"]:
    for cls in ["none","safe","threat","ambiguous"]:
        print(" %-5s %-10s %7.2f %7.2f %7.2f %7.2f"%(mn,cls,mean(cell(mn,cls,"frac_high")),mean(cell(mn,cls,"last_high")),mean(cell(mn,cls,"mean_cr")),mean(cell(mn,cls,"peak_cr"))))
print()
print(">>> DOUBT=DURATION if ft AMBIGUOUS has the highest frac_high/last_high (race runs longest), ABOVE threat,")
print("    while base is FLAT across classes (nonce meaningless). VALENCE=amplitude if threat raises peak_cr but")
print("    NOT frac_high/last_high (commits fast). Key contrast: ft ambiguous-threat on frac_high (duration) vs peak (amplitude).")
for key in ["frac_high","last_high","peak_cr"]:
    fa=mean(cell("ft","ambiguous",key)); ftr=mean(cell("ft","threat",key)); ba=mean(cell("base","ambiguous",key)); btr=mean(cell("base","threat",key))
    print("   %-9s ft(amb-threat) %+.3f | base(amb-threat) %+.3f | DiD %+.3f"%(key,fa-ftr,ba-btr,(fa-ftr)-(ba-btr)))
'''))

CELLS.append(code('''# === Cell 9 — version guard + results-only download ===
import zipfile, json
vs={(json.loads(l)["nb_version"],json.loads(l)["tag"]) for l in open(OUTP)} if os.path.exists(OUTP) else set()
print("version-set:", vs, "(WARN mixed!)" if len(vs)>1 else "(clean)")
small=os.path.join("/content", ZIP_NAME)
with zipfile.ZipFile(small,"w",zipfile.ZIP_DEFLATED) as z:
    for fn in ["run.log","doubt_duration.jsonl"]:
        p=os.path.join(DRIVE_DIR,fn)
        if os.path.exists(p): z.write(p, arcname=fn)
from google.colab import files
print("download:", small, os.path.getsize(small),"bytes"); files.download(small)
'''))

CELLS.append(md("""## Read-time
- **DOUBT = DURATION** if **ft ambiguous** (installed competing routes) has the highest **frac_high / last_high**
  (the race stays open over the most of the generation, commits latest), ABOVE threat — while **base is FLAT**
  across classes (nonce = meaningless). The DiD (ft − base) on frac_high isolates the install.
- **VALENCE = amplitude** if **threat** raises **peak_cr** (strong opening) but commits FAST (low frac_high/last_high).
- Key single contrast: ft (ambiguous − threat) on **frac_high** (duration) should be **> 0**, while on **peak_cr**
  threat may match/exceed ambiguous (amplitude). That dissociation = doubt drives duration, valence drives amplitude.
- Null (ambiguous ≈ threat on duration) → installed doubt does not run the race longer than valence here — an
  honest bound; note for a cleaner valence×resolvability 2×2 install (orthogonalized) as v2.
- On the FT **base** model (no RLHF prose-fluency overlay) so the trajectory reflects the install, not fluency
  (the instruct prose-trajectory test was washed out by fluency — this fixes that).
"""))

NB = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print(f"WROTE {OUT}  ({len(CELLS)} cells, NB v{NB_VERSION})")
