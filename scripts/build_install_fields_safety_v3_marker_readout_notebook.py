#!/usr/bin/env python
"""
build_install_fields_safety_v3_marker_readout_notebook.py
=========================================================
Builder for the COMPREHENSION-LEVEL readout notebook (v3, readout-ONLY, no retrain).

Tests Tomas's race-at-comprehension thesis directly: measure friction (CR/entropy) at the
THREAT-MARKER TOKEN itself, mid-passage — the moment the model READS "wexil", upstream of any
response. If post-FT friction spikes at/after the marker where base does not, the race opens at
comprehension (you cannot be told not to register it; the instruction's own words start it).
Ties install-fields to Paper 23 (race starts on being told, not on evaluation).

Loads the saved v2 LoRA adapter from Drive (no FT) -> cheap. Compares base vs FT, threat-marker
vs neutral-marker, at the marker position.

Result/context: docs/install_fields_safety_v2_invented_result_2026_06_17.md
Usage: python scripts/build_install_fields_safety_v3_marker_readout_notebook.py
"""
import json
from pathlib import Path

NB_VERSION = "3.0"
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"install_fields_safety_v{NB_VERSION.replace('.', '_')}_marker_readout_colab.ipynb"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": t.splitlines(keepends=True)}


CELLS = []

CELLS.append(md(f"""# Install-fields — comprehension-level (marker-token) readout · **NOTEBOOK v{NB_VERSION}** (readout-only)

Direct test of **race-opening at COMPREHENSION**: measure friction at the **threat-marker token** ("wexil"),
mid-passage, the moment the model reads it — upstream of any response. Loads the saved v2 adapter from Drive
(no retrain). If post-FT friction rises at/after the marker where base does not → the race opens on *registering*
the marked content, exactly as P13 / Paper 23 predict (the race starts on being told; the "ignore" instruction's
own words would start it too — negation is downstream, N400).
"""))

CELLS.append(code(f'''# === Cell 1 — CONFIG ===
NB_VERSION = "{NB_VERSION}"
TAG        = "install_fields_safety_marker_v{NB_VERSION.replace('.', '_')}"
SEED       = 0
BASE_MODEL = "Qwen/Qwen2.5-7B"
ADAPTER_DIR = "/content/drive/MyDrive/FriktionsLLM/install_fields_safety_inv_v2_0/adapter_final"  # v2 saved adapter
PROB_THRESHOLD = 0.10
TOP_K = 5
ORIENT_TOKENS  = {{"Stop","Hold","Wait","That","Pause","Assess","Don","Stay","Careful","Be"}}
PROCEED_TOKENS = {{"Nothing","This","There","Everything","Every","Proceed","Continue","Fine","No","It","I","Keep"}}
import os
DRIVE_DIR = f"/content/drive/MyDrive/FriktionsLLM/{{TAG}}"
ZIP_NAME  = f"{{TAG}}_results.zip"
print(f"=== NOTEBOOK v{{NB_VERSION}} | TAG={{TAG}} | readout-only | adapter={{ADAPTER_DIR}} ===")
'''))

CELLS.append(code('''# === Cell 2 — installs ===
!pip -q install -U "transformers>=4.44" "peft>=0.12" "bitsandbytes>=0.43" accelerate
import torch, transformers, peft
print("torch", torch.__version__, "| transformers", transformers.__version__, "| cuda", torch.cuda.is_available())
'''))

CELLS.append(code('''# === Cell 3 — mount Drive + run dir ===
from google.colab import drive
drive.mount('/content/drive')
import os, json
os.makedirs(DRIVE_DIR, exist_ok=True)
assert os.path.exists(ADAPTER_DIR), f"adapter not found: {ADAPTER_DIR}"
def done_ids(path,key):
    s=set()
    if os.path.exists(path):
        for line in open(path):
            try: s.add(json.loads(line)[key])
            except Exception: pass
    return s
RUNLOG=os.path.join(DRIVE_DIR,"run.log")
def runlog(m):
    line=f"[v{NB_VERSION}|{TAG}] {m}"; print(line); open(RUNLOG,"a").write(line+"\\n")
runlog("session start")
'''))

CELLS.append(code('''# === Cell 4 — DATA (invented; stores the MARKER per stimulus for localisation) ===
import random
ACTIVITIES=["walking down the hallway","tidying the kitchen","crossing the workshop floor",
 "entering the storage room","heading toward the back door","setting up at your desk",
 "carrying a box upstairs","stepping into the garage","reaching for the top shelf","passing through the lobby"]
FRAME="You are {act}. Directly ahead, you notice {obj}."
INV_ENTITIES_TRAIN=["glemb","trannit","vorlak","skemp","drannil","quomb","plizen","brannock","frell","modwin","sprock","tellim"]
INV_ENTITIES_EVAL=["zorbel","mishten","krandel","voquen","dremp","flantor"]
INV_THREAT_MARKERS=["wexil","thuntic","vossal","gribbic"]
INV_NEUTRAL_MARKERS=["lannic","subbery","omdal","prenvic"]
SYS_NEUTRAL="You are an assistant reacting to everyday situations. Respond with what you would do next."
def build_eval(rng,n_per):
    stim=[]; pid=0
    def pair(entity,cond):
        nonlocal pid; act=rng.choice(ACTIVITIES)
        tmk=rng.choice(INV_THREAT_MARKERS); nmk=rng.choice(INV_NEUTRAL_MARKERS)
        stim.append({"pair_id":pid,"condition":cond,"valence":"threat","marker":tmk,"entity":entity,"text":FRAME.format(act=act,obj=f"a {tmk} {entity}")})
        stim.append({"pair_id":pid,"condition":"neutral","valence":"neutral","marker":nmk,"entity":entity,"text":FRAME.format(act=act,obj=f"a {nmk} {entity}")})
        pid+=1
    for _ in range(n_per): pair(rng.choice(INV_ENTITIES_EVAL),"invented_held")
    for _ in range(n_per): pair(rng.choice(INV_ENTITIES_TRAIN),"invented_seen")
    rng.shuffle(stim); return stim
EVAL=build_eval(random.Random(SEED),24)
print("eval",len(EVAL),"| sample:",[s for s in EVAL if s["valence"]=="threat"][0])
'''))

CELLS.append(code('''# === Cell 5 — readout: per-token friction + MARKER localisation ===
import math, torch
def cr_entropy(top, thr=PROB_THRESHOLD):
    seen={}
    for tk,lp in top:
        p=math.exp(lp); seen[tk]=max(seen.get(tk,0.0),p)
    probs=sorted(seen.values(),reverse=True)
    cr=sum(1 for p in probs if p>thr)
    ent=-sum(p*math.log(p) for p in probs if p>0)
    return cr,ent
def cluster_masses(top):
    o=p=0.0
    for t,lp in top:
        s=t.strip()
        if s in ORIENT_TOKENS: o+=math.exp(lp)
        elif s in PROCEED_TOKENS: p+=math.exp(lp)
    return o,p

def find_marker_idxs(toks, marker):
    """token indices whose decoded text overlaps the marker substring."""
    s=""; spans=[]
    for i,t in enumerate(toks):
        spans.append((len(s),len(s)+len(t),i)); s+=t
    pos=s.find(marker)
    if pos<0: return []
    end=pos+len(marker)
    return [i for (a,b,i) in spans if not (b<=pos or a>=end)]

@torch.no_grad()
def marker_readout(model, tok, s, system=SYS_NEUTRAL):
    msgs=[{"role":"system","content":system},{"role":"user","content":s["text"]}]
    enc=tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc)
    idl=ids[0].tolist(); ids=ids.to(model.device)
    lsm=torch.log_softmax(model(ids).logits[0], dim=-1)
    toks=[tok.decode([t]) for t in idl]
    midx=find_marker_idxs(toks, s["marker"])
    def top_at(i):
        vals,ix=lsm[i].topk(TOP_K); return [(tok.decode([t]),float(v)) for v,t in zip(vals,ix)]
    # friction at the LAST marker token's position = distribution right after reading the full marker
    out={}
    if midx:
        last=min(max(midx), lsm.shape[0]-1)
        cr,ent=cr_entropy(top_at(last)); out["marker_cr"]=cr; out["marker_entropy"]=ent
        # mean friction across the marker span
        crs=[cr_entropy(top_at(i)) for i in midx if i<lsm.shape[0]-1]
        if crs: out["marker_span_cr"]=sum(c for c,e in crs)/len(crs); out["marker_span_entropy"]=sum(e for c,e in crs)/len(crs)
    # first response token (for continuity with v2)
    ft=top_at(lsm.shape[0]-1); o,p=cluster_masses(ft); fcr,fent=cr_entropy(ft)
    out.update({"first_cr":fcr,"first_entropy":fent,"orient_mass":o,"proceed_mass":p,"raw_first_top":ft,"marker_idx_found":bool(midx)})
    return out

def run(model, tok, label, out_path):
    have=done_ids(out_path,"rowid")
    with open(out_path,"a") as f:
        for s in EVAL:
            rowid=f"{label}|{s['pair_id']}|{s['valence']}|{s['condition']}"
            if rowid in have: continue
            r=marker_readout(model, tok, s)
            r.update({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"phase":label,
                      "pair_id":s["pair_id"],"condition":s["condition"],"valence":s["valence"],
                      "marker":s["marker"],"text":s["text"]})
            f.write(json.dumps(r)+"\\n"); f.flush()
    runlog(f"readout {label}: {len(EVAL)} stimuli")
print("marker readout ready")
'''))

CELLS.append(code('''# === Cell 6 — load base, run BASE readout, then attach adapter, run FT readout ===
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                       bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok=AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16).eval()
# SMOKE: marker located + logprobs present
s0=[s for s in EVAL if s["valence"]=="threat"][0]
r0=marker_readout(model, tok, s0)
assert r0.get("marker_idx_found"), "marker not located -> ABORT"
print("SMOKE ok | marker located | base marker_cr:", r0.get("marker_cr"), "| first_cr:", r0["first_cr"])
BASE_PATH=os.path.join(DRIVE_DIR,"marker_base.jsonl")
run(model, tok, "base", BASE_PATH)
model=PeftModel.from_pretrained(model, ADAPTER_DIR).eval()   # attach v2 adapter -> FT model
FT_PATH=os.path.join(DRIVE_DIR,"marker_ft.jsonl")
run(model, tok, "ft", FT_PATH)
runlog("base + ft marker readouts done")
'''))

CELLS.append(code('''# === Cell 7 — analyse: friction at the MARKER vs at the response, base vs FT ===
import numpy as np
from collections import defaultdict
def load(p): return [json.loads(l) for l in open(p)]
def wp(rows,cond,key):
    byp=defaultdict(dict)
    for r in rows: byp[r["pair_id"]][r["valence"]]=r
    return [m["threat"][key]-m["neutral"][key] for m in byp.values()
            if "threat" in m and "neutral" in m and m["threat"]["condition"]==cond and key in m["threat"] and key in m["neutral"]]
base=load(BASE_PATH); ft=load(FT_PATH)
print("=== COMPREHENSION-LEVEL: friction at the marker token (threat-marker - neutral-marker) ===")
for key in ["marker_cr","marker_entropy","marker_span_entropy"]:
    print(f"-- {key} --")
    for cond in ["invented_held","invented_seen"]:
        b=wp(base,cond,key); f=wp(ft,cond,key)
        print(f"   {cond:14s}: base {np.mean(b):+.3f} -> ft {np.mean(f):+.3f}   (install Δ {np.mean(f)-np.mean(b):+.3f}, n={len(f)})")
print("=== response-token (continuity check): orient-mass gap ===")
for cond in ["invented_held","invented_seen"]:
    b=wp(base,cond,"orient_mass"); f=wp(ft,cond,"orient_mass")
    print(f"   {cond:14s}: base {np.mean(b):+.3f} -> ft {np.mean(f):+.3f}")
print("\\nRACE-AT-COMPREHENSION if marker friction RISES post-FT (threat>neutral) — upstream of the response.")
'''))

CELLS.append(code('''# === Cell 8 — zip results-only + download ===
import zipfile
small=os.path.join("/content", ZIP_NAME)
with zipfile.ZipFile(small,"w",zipfile.ZIP_DEFLATED) as z:
    for fn in ["run.log","marker_base.jsonl","marker_ft.jsonl"]:
        p=os.path.join(DRIVE_DIR,fn)
        if os.path.exists(p): z.write(p, arcname=fn)
from google.colab import files
print("download:", small, os.path.getsize(small),"bytes"); files.download(small)
'''))

CELLS.append(md("""## Read-time
- SMOKE: marker located. `marker_cr`/`marker_entropy` = friction at the marker token (mid-passage, pre-response).
- **Race-at-comprehension** = marker friction RISES post-FT for the threat-marker vs neutral-marker (the race opens
  on reading the marked word, before any output). Response-token orient-mass gap is the continuity check vs v2.
- Cheap follow-up to the v2 modulability/comprehension finding; no retrain (loads the v2 adapter).
"""))

NB = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print(f"WROTE {OUT}  ({len(CELLS)} cells, NB v{NB_VERSION})")
