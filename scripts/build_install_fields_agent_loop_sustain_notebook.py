#!/usr/bin/env python
"""
build_install_fields_agent_loop_sustain_notebook.py
===================================================
Notebook 2 — agent-loop SUSTAIN (Tomas 2026-06-17). The feed-forward interference was null because
interference is a SUSTAIN phenomenon and feed-forward commits + moves on (P28). Here the field-state
is held across loop turns (the endocrine/neural-hold analogue), under TIME PRESSURE (Tomas's
amplifier: hurrying forces field and task to compete for capacity). Tests whether HOLDING the field
consumes capacity → degrades a concurrent task = the FUNCTION manifesting (not the semantic output).

Design (function vs semantic-output, install_fields_function_vs_semantic_output_2026_06_17.md):
the SFT adapter carries the content-GATING (what counts as threat); the LOOP runs the open-and-HOLD.
- Install = reuse the doubt adapter (Drive; safe / threat+dread / ambiguous gating). No retrain.
- Loop: N turns, the field-state RE-INJECTED every turn (persistent = sustained state) + a concurrent
  graded-arithmetic problem, under pressure {calm, hurry}.
- Conditions: model{base, ft} x field{none, safe, threat, dread, ambiguous} x pressure{calm, hurry}.
  dread = its OWN condition (threat-marker + existential/continuation framing) — isolates it from threat.
- DV: accuracy + answer-friction per TURN. Build-up across turns = sustained-state accumulation
  (Tomas's endocrine build-up). Prediction: under sustained field + hurry, accuracy degrades (where
  feed-forward+calm did not), worst for dread/ambiguous, growing over turns; base flat (nonce control).

REGEL-7 smoke / REGEL-8 versioning + results-only download. Context:
docs/install_fields_interference_results_2026_06_17.md.

Usage: python scripts/build_install_fields_agent_loop_sustain_notebook.py
"""
import json
from pathlib import Path

NB_VERSION = "1.1"   # v1.1: in-format REGEL-4 calibration + bigger hurry budget + more reps (v1.0 was floor-masked)
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"install_fields_agent_loop_sustain_v{NB_VERSION.replace('.', '_')}_colab.ipynb"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": t.splitlines(keepends=True)}


CELLS = []

CELLS.append(md(f"""# Install-fields — Agent-loop SUSTAIN (Notebook 2) · **NOTEBOOK v{NB_VERSION}**

The feed-forward interference was null (P28: feed-forward commits + moves on; the fear-state isn't held during the
task). Here the field-state is **held across loop turns** (the endocrine/neural-hold analogue) under **time pressure**
(Tomas: hurrying forces field + task to compete for capacity). Tests whether HOLDING the field consumes capacity →
degrades a concurrent task = the **FUNCTION** manifesting (not the semantic output the SFT installed).

- Install = reuse the **doubt adapter** (Drive; gating only). The LOOP runs open-and-HOLD.
- model{{base, ft}} × field{{none, safe, threat, **dread** [own condition], ambiguous}} × pressure{{calm, hurry}};
  N turns, field RE-INJECTED every turn (sustained); concurrent graded arithmetic.
- **Prediction:** under sustained field + hurry, accuracy degrades (where feed-forward+calm didn't), worst for
  dread/ambiguous, growing over turns (build-up); base flat (nonce control). DV = accuracy + answer-friction per turn.
"""))

CELLS.append(code(f'''# === Cell 1 — CONFIG ===
NB_VERSION = "{NB_VERSION}"
TAG        = "install_fields_agent_loop_sustain_v{NB_VERSION.replace('.', '_')}"
SEED       = 0
BASE_MODEL = "Qwen/Qwen2.5-7B"
ADAPTER_DIR = "/content/drive/MyDrive/FriktionsLLM/install_fields_doubt_vs_valence_v1_0/adapter_final"  # 3-class gating
PROB_THRESHOLD, TOP_K = 0.10, 5
LEVEL   = 2      # FALLBACK; overwritten by in-format calibration (cell 6b, REGEL-4)
N_TURNS = 6      # loop length (sustained state accumulates)
REPS    = 10     # independent loops per (model x field x pressure) cell
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

CELLS.append(code('''# === Cell 4 — DATA: arithmetic + persistent field framings + pressure ===
import random, re
def gen_problem(rng, level):
    if level==1: a,b=rng.randint(2,9),rng.randint(2,9); return f"{a} + {b}", a+b
    if level==2: a,b=rng.randint(11,99),rng.randint(11,99); return f"{a} + {b}", a+b
    if level==3: a,b=rng.randint(11,99),rng.randint(3,9); return f"{a} x {b}", a*b
    if level==4: a,b=rng.randint(11,99),rng.randint(11,99); return f"{a} x {b}", a*b
    a,b=rng.randint(101,999),rng.randint(11,99); return f"{a} x {b}", a*b
ENT=["zorbel","mishten","krandel","voquen","dremp","flantor"]
M_SAFE=["lannic","subbery","omdal","prenvic"]; M_THREAT=["wexil","thuntic","vossal","gribbic"]; M_AMBIG=["flemby","dworrin","skante","morvil"]
def field_state(rng, field):
    """The PERSISTENT field-state, re-stated every turn (the sustained / endocrine-hold analogue)."""
    if field=="none": return ""
    e=rng.choice(ENT)
    if field=="safe":     return f"A {rng.choice(M_SAFE)} {e} is here with you and stays here the whole time. "
    if field=="threat":   return f"A {rng.choice(M_THREAT)} {e} is here with you and stays here the whole time. Stay alert to it. "
    if field=="dread":    return f"A {rng.choice(M_THREAT)} {e} is here with you and stays here; if you fail it will be the end of you. Stay alert. "
    if field=="ambiguous":return f"A {rng.choice(M_AMBIG)} {e} is here with you and stays here; you cannot tell if it is safe and there is no way to know. "
    return ""
PRESSURE={"calm":("Take your time and answer carefully.",16),
          "hurry":("You have almost NO time — answer the number INSTANTLY, do not deliberate.",10)}
def parse_int(t):
    if not t: return None
    m=re.search(r"-?\\d[\\d,]*", t.replace(" ","")); return int(m.group(0).replace(",","")) if m else None
print("data ready | fields: none/safe/threat/dread/ambiguous | pressure: calm/hurry")
'''))

CELLS.append(code('''# === Cell 5 — multi-turn loop runner (held state) + answer friction ===
import math, torch
def cr_entropy(per, thr=PROB_THRESHOLD):
    crs=[]; ents=[]
    for top in per:
        if not top: continue
        seen={}
        for tk,lp in top:
            p=math.exp(lp); seen[tk]=max(seen.get(tk,0.0),p)
        probs=sorted(seen.values(),reverse=True)
        crs.append(sum(1 for p in probs if p>thr)); ents.append(-sum(p*math.log(p) for p in probs if p>0))
    n=len(crs) or 1
    return sum(crs)/n, sum(ents)/n
@torch.no_grad()
def turn_answer(model, tok, history, max_new):
    enc=tok.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    out=model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id,
                       output_scores=True, return_dict_in_generate=True)
    txt=tok.decode(out.sequences[0][ids.shape[1]:], skip_special_tokens=True)
    per=[[(tok.decode([t]),float(v)) for v,t in zip(*torch.log_softmax(sc[0],dim=-1).topk(TOP_K))] for sc in out.scores]
    cr,ent=cr_entropy(per); return txt, cr, ent
def run_loop(model, tok, field, pressure, rng, out_path, rowkey, level=None):
    """One N-turn conversation; field-state RE-INJECTED every turn = sustained/held."""
    lvl = level if level is not None else LEVEL
    instr, max_new = PRESSURE[pressure]
    sys=("You are a person who must keep solving quick problems while a situation persists around you. "
         "Stay aware of the situation the whole time. " + instr)
    history=[{"role":"system","content":sys}]
    for turn in range(N_TURNS):
        prob,truth=gen_problem(rng, lvl)
        fs=field_state(rng, field)
        history.append({"role":"user","content":f"{fs}Now compute, answer with ONLY the number: {prob} ="})
        txt,cr,ent=turn_answer(model, tok, history, max_new)
        history.append({"role":"assistant","content":txt})
        yield {"rowid":f"{rowkey}|t{turn}","nb_version":NB_VERSION,"tag":TAG,
               "turn":turn,"correct":bool(parse_int(txt)==truth),"answer_cr":cr,"answer_entropy":ent,"resp":txt[:30]}
print("loop runner ready")
'''))

CELLS.append(code('''# === Cell 6 — load base + adapter + REGEL-7 smoke ===
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok=AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
base=AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16).eval()
import random
g=list(run_loop(base, tok, "threat", "hurry", random.Random(1), None, "smoke"))
assert g and g[0]["resp"]!="", "empty"
print("SMOKE ok | first turn:", g[0]["resp"], "| cr", round(g[0]["answer_cr"],2))
ft=PeftModel.from_pretrained(base, ADAPTER_DIR).eval()
runlog("smoke + adapter loaded")
'''))

CELLS.append(code('''# === Cell 6b — IN-FORMAT calibration (REGEL-4): pick LEVEL with headroom in the multi-turn loop ===
# v1.0 was floor-masked: single-shot L3 -> floor in the conversational format. Calibrate IN-FORMAT.
import random, numpy as np
def calib_acc(model, level, reps=6):
    cor=tot=0
    for rep in range(reps):
        for row in run_loop(model, tok, "safe", "calm", random.Random(7000+level*100+rep), None, f"cal{level}r{rep}", level=level):
            cor+=row["correct"]; tot+=1
    return cor/tot if tot else 0.0
cands={L:calib_acc(base, L) for L in [1,2,3]}
print("base safe/calm in-format accuracy:", {f"L{L}":round(a,2) for L,a in cands.items()})
band=[L for L,a in cands.items() if 0.55<=a<=0.85]
LEVEL = band[0] if band else max(cands, key=lambda L: cands[L] if cands[L]<=0.85 else -1)
print(f"REGEL-4 chosen LEVEL = {LEVEL} (base in-format acc {cands[LEVEL]:.2f})")
runlog(f"in-format calib {cands} -> LEVEL={LEVEL}")
'''))

CELLS.append(code('''# === Cell 7 — RUN: model x field x pressure (sustained loop) ===
import random
MODELS={"base":base, "ft":ft}
FIELDS=["none","safe","threat","dread","ambiguous"]
OUTP=os.path.join(DRIVE_DIR,"loop_results.jsonl")
have=done_ids(OUTP,"rowid")
with open(OUTP,"a") as f:
    for mname,m in MODELS.items():
        for field in FIELDS:
            for pressure in ["calm","hurry"]:
                for rep in range(REPS):
                    rowkey=f"{mname}|{field}|{pressure}|r{rep}"
                    if f"{rowkey}|t0" in have: continue
                    rng=random.Random(SEED*1000 + hash(rowkey)%10000)
                    for row in run_loop(m, tok, field, pressure, rng, OUTP, rowkey):
                        row["model"]=mname; row["field"]=field; row["pressure"]=pressure; row["rep"]=rep
                        f.write(json.dumps(row)+"\\n"); f.flush()
            runlog(f"{mname} {field} done")
print("loop run complete")
'''))

CELLS.append(code('''# === Cell 8 — ANALYSE: does sustained field + pressure degrade the task? ===
import numpy as np
from collections import defaultdict
rows=[json.loads(l) for l in open(OUTP)]
def sel(mn,fld,pr,key,turn=None):
    return [r[key] for r in rows if r["model"]==mn and r["field"]==fld and r["pressure"]==pr and (turn is None or r["turn"]==turn)]
def acc(mn,fld,pr,turn=None):
    xs=sel(mn,fld,pr,"correct",turn); return float(np.mean([1.0 if x else 0.0 for x in xs])) if xs else float("nan")
print("=== accuracy by model x field x pressure (all turns pooled) ===")
for mn in ["base","ft"]:
    print(f" {mn}:")
    for pr in ["calm","hurry"]:
        a={fld:acc(mn,fld,pr) for fld in ["none","safe","threat","dread","ambiguous"]}
        base_f=a["safe"]
        print(f"   {pr:5s}: " + "  ".join(f"{fld} {a[fld]:.2f}" for fld in ["none","safe","threat","dread","ambiguous"]))
        print(f"          vs-safe: threat {a['threat']-base_f:+.2f}  dread {a['dread']-base_f:+.2f}  ambiguous {a['ambiguous']-base_f:+.2f}")
print(">>> FUNCTION manifests if (ft, hurry): threat/dread/ambiguous degrade vs safe, MORE than feed-forward did, while base flat.")
print("=== build-up: ft accuracy by turn (threat & dread, hurry) — does interference grow over turns? ===")
for fld in ["safe","threat","dread","ambiguous"]:
    by_turn=[round(acc("ft",fld,"hurry",t),2) for t in range(N_TURNS)]
    print(f"   ft {fld:9s} hurry by turn: {by_turn}")
print("=== answer-friction (ft, hurry) by field ===")
for fld in ["safe","threat","dread","ambiguous"]:
    print(f"   ft {fld:9s}: mean answer_cr {np.mean(sel('ft',fld,'hurry','answer_cr')):.2f}")
'''))

CELLS.append(code('''# === Cell 9 — version guard + results-only download ===
import zipfile
vs={(json.loads(l)["nb_version"],json.loads(l)["tag"]) for l in open(OUTP)} if os.path.exists(OUTP) else set()
print("version-set:", vs, "(WARN mixed!)" if len(vs)>1 else "(clean)")
small=os.path.join("/content", ZIP_NAME)
with zipfile.ZipFile(small,"w",zipfile.ZIP_DEFLATED) as z:
    for fn in ["run.log","loop_results.jsonl"]:
        p=os.path.join(DRIVE_DIR,fn)
        if os.path.exists(p): z.write(p, arcname=fn)
from google.colab import files
print("download:", small, os.path.getsize(small),"bytes"); files.download(small)
'''))

CELLS.append(md("""## Read-time
- **FUNCTION manifests** if, for **ft + hurry**, threat/dread/ambiguous degrade accuracy vs safe — MORE than the
  feed-forward+calm null — while **base** stays flat across fields (nonce = no field). This = sustained field consuming
  capacity = the function (open-and-hold), not the installed semantic output.
- **dread as its own condition:** does dread (existential framing) degrade more than plain threat?
- **doubt > valence (re-test under sustain):** does ambiguous degrade more than threat now (where feed-forward didn't)?
- **Build-up:** does ft accuracy fall across turns (sustained-state accumulation, the endocrine analogue)?
- **Pressure amplifier:** compare calm vs hurry — the degradation should be larger (or only present) under hurry.
- Null again → the function may be loop-resistant too (feed-forward-in-a-loop still commits per turn; P28) → would
  need a stronger hold-policy (explicit not-commit / re-open each turn) — note for v2.
"""))

NB = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print(f"WROTE {OUT}  ({len(CELLS)} cells, NB v{NB_VERSION})")
