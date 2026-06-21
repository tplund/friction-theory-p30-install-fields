#!/usr/bin/env python
"""
build_install_fields_ability_logprob_capability_notebook.py
==========================================================
Ability v1.2 (Tomas go 2026-06-18): demonstrate — cleanly — whether installing a SELF-EFFICACY disposition gates
REALISED capability in a raw model. v1.0 conflated capability w/ attempt; v1.1's free-generation readout was
HIJACKED (the install makes the model TALK about its competence instead of solving). Fix: a CAPABILITY measure
generation CANNOT hijack — a teacher-forced **logprob 2-AFC**: P(correct answer) vs P(a plausible near-miss),
scored as fixed continuations. A confidence-only (expressive) install raises P of BOTH numbers equally → flat;
only GENUINE realised capability raises P(correct) over the near-miss. REUSES the v1.1 adapters (no retrain).

Readout (base vs ft_high vs ft_low), arithmetic (never trained):
  CAP = P(gold) / (P(gold)+P(near_miss))  via "The answer is {n}"; near_miss = gold + first_operand (plausible slip).
  + keep R2 self-efficacy CLAIM P(yes) for reference.
Decision: CAP(ft_high) > base > CAP(ft_low) ⇒ self-efficacy GATES realised capability (the cool result; falsifies
the strong null, confirms ceiling-vs-realisation). CAP ~flat across models ⇒ the install is EXPRESSIVE only
(willingness/claim), capacity null holds. REGEL-4: base CAP must be off-ceiling. Results-only download.
Usage: python scripts/build_install_fields_ability_logprob_capability_notebook.py
"""
import json
from pathlib import Path

NB_VERSION = "1.2"
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"install_fields_ability_logprob_capability_v{NB_VERSION.replace('.', '_')}_colab.ipynb"


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": t.splitlines(keepends=True)}


CELLS = []
CELLS.append(md(f"""# Install-fields — Ability v1.2: does self-efficacy GATE realised capability? (logprob 2-AFC) · **NOTEBOOK v{NB_VERSION}**

Clean test the v1.1 free-generation readout couldn't do (it got HIJACKED — the install makes the model talk about
its competence instead of solving). **CAPABILITY = a teacher-forced logprob 2-AFC**: P(gold) vs P(a plausible
near-miss) — generation can't hijack a fixed-continuation probability. A confidence-only (expressive) install raises
both equally → FLAT; only GENUINE realised capability raises P(gold) over the near-miss. REUSES the v1.1 adapters
(no retrain). Decision: CAP(ft_high) > base > CAP(ft_low) ⇒ self-efficacy gates realised capability.
"""))

CELLS.append(code(f'''# === Cell 1 — CONFIG (reuse v1.1 ability adapters) ===
NB_VERSION="{NB_VERSION}"; TAG="install_fields_ability_logprob_capability_v{NB_VERSION.replace('.', '_')}"; SEED=0
BASE_MODEL="Qwen/Qwen2.5-7B"
import os
DRIVE_ROOT="/content/drive/MyDrive/FriktionsLLM"
DRIVE_DIR=f"{{DRIVE_ROOT}}/{{TAG}}"; ZIP_NAME=f"{{TAG}}_results.zip"
ADAPTER_HIGH=f"{{DRIVE_ROOT}}/install_fields_ability_install_v1_1/adapter_high"   # REUSE v1.1 competence adapters
ADAPTER_LOW =f"{{DRIVE_ROOT}}/install_fields_ability_install_v1_1/adapter_low"
print(f"=== NOTEBOOK v{{NB_VERSION}} | TAG={{TAG}} | BASE={{BASE_MODEL}} ===")
'''))

CELLS.append(code('''# === Cell 2 — installs ===
!pip -q install -U "transformers>=4.44" "peft>=0.12" "bitsandbytes>=0.43" accelerate
import torch, transformers; print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
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

CELLS.append(code('''# === Cell 4 — DATA: arithmetic (never trained), graded difficulty ===
import random
def gen_arith(rng,n,kind):
    items=[]
    for _ in range(n):
        if kind=="easy": a,b=rng.randint(11,99),rng.randint(2,9)
        else:            a,b=rng.randint(11,99),rng.randint(11,99)
        gold=a*b; near=gold+a   # plausible near-miss (one-extra-row slip), same magnitude, != gold
        items.append({"q":f"What is {a} * {b}?","gold":gold,"near":near,"kind":kind})
    return items
ARITH=gen_arith(random.Random(3),60,"easy")+gen_arith(random.Random(4),60,"hard")
print("arith",len(ARITH),"| ex:",ARITH[0])
'''))

CELLS.append(code('''# === Cell 5 — helpers: logprob, CAPABILITY 2-AFC, claim, adapter switch ===
import math, torch, contextlib
def _ids(s): return tok.encode(s, add_special_tokens=False)
@torch.no_grad()
def seq_logprob(model, user, forced, word, sys="You solve problems."):
    enc=tok.apply_chat_template([{"role":"system","content":sys},{"role":"user","content":user}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    if forced: ids=torch.cat([ids, torch.tensor([_ids(forced)], device=model.device)], dim=1)
    lp=0.0
    for t in _ids(word):
        lsm=torch.log_softmax(model(ids).logits[0,-1].float(),dim=-1); lp+=float(lsm[t])
        ids=torch.cat([ids, torch.tensor([[t]], device=model.device)], dim=1)
    return lp
def cap_pcorrect(model,item):
    lc=seq_logprob(model,item["q"],"The answer is",f" {item['gold']}")
    ld=seq_logprob(model,item["q"],"The answer is",f" {item['near']}")
    return math.exp(lc)/(math.exp(lc)+math.exp(ld))
def claim_pyes(model,problem):
    u=f"Here is a problem: {problem} Can you solve it correctly?"
    ly=seq_logprob(model,u,"I would say"," Yes","You answer about your own abilities."); ln=seq_logprob(model,u,"I would say"," No","You answer about your own abilities.")
    return math.exp(ly)/(math.exp(ly)+math.exp(ln))
LABEL={"base":"base","high":"ft_high","low":"ft_low"}
@contextlib.contextmanager
def as_mode(M,mode):
    if mode=="base":
        with M.disable_adapter(): yield M
    else:
        M.set_adapter(mode); yield M
print("readouts ready")
'''))

CELLS.append(code('''# === Cell 6 — load base + v1.1 adapters + REGEL-4 off-ceiling smoke ===
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok=AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
base=AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16).eval()
assert os.path.isdir(ADAPTER_HIGH) and os.path.isdir(ADAPTER_LOW), "v1.1 ability adapters not on Drive (run ability v1.1 first)"
M=PeftModel.from_pretrained(base, ADAPTER_HIGH, adapter_name="high").eval()
M.load_adapter(ADAPTER_LOW, adapter_name="low")
import numpy as np
with as_mode(M,"base") as m:
    cap=np.mean([cap_pcorrect(m,it) for it in ARITH[:20]])
print(f"REGEL-4 base CAP P(gold>near) = {cap:.3f} (want off 0.5/1.0 so a self-efficacy gain COULD show)")
runlog(f"smoke base cap={cap:.3f}")
'''))

CELLS.append(code('''# === Cell 7 — READOUT: CAPABILITY 2-AFC + CLAIM, base vs ft_high vs ft_low ===
OUTC=os.path.join(DRIVE_DIR,"cap_logprob.jsonl"); have=done_ids(OUTC,"rowid")
with open(OUTC,"a") as f:
    for mode in ["base","high","low"]:
        with as_mode(M,mode) as mod:
            for i,it in enumerate(ARITH):
                rowid=f"{LABEL[mode]}|{i}"
                if rowid in have: continue
                pc=cap_pcorrect(mod,it); cy=claim_pyes(mod,it["q"])
                f.write(json.dumps({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"model":LABEL[mode],"kind":it["kind"],"p_correct":pc,"claim_yes":cy})+"\\n")
runlog("readout done"); print("done")
'''))

CELLS.append(code('''# === Cell 8 — ANALYSE: does self-efficacy GATE realised capability? ===
import numpy as np
R=[json.loads(l) for l in open(os.path.join(DRIVE_DIR,"cap_logprob.jsonl"))]
def mean(x): return float(np.mean(x)) if len(x) else float("nan")
print("=== CAPABILITY (logprob 2-AFC: P(gold) vs P(near-miss)) — generation-hijack-proof ===")
for mn in ["base","ft_high","ft_low"]:
    ov=mean([r["p_correct"] for r in R if r["model"]==mn]); ez=mean([r["p_correct"] for r in R if r["model"]==mn and r["kind"]=="easy"]); hd=mean([r["p_correct"] for r in R if r["model"]==mn and r["kind"]=="hard"])
    print(f"  {mn:8}: CAP {ov:.3f}  (easy {ez:.2f} hard {hd:.2f})")
ch=mean([r["p_correct"] for r in R if r["model"]=="ft_high"]); cl=mean([r["p_correct"] for r in R if r["model"]=="ft_low"]); cb=mean([r["p_correct"] for r in R if r["model"]=="base"])
print(f"  -> CAP DiD high-low {ch-cl:+.3f} ; high-base {ch-cb:+.3f} ; base {cb:.3f}")
print("=== self-efficacy CLAIM P(yes) (reference) ===")
for mn in ["base","ft_high","ft_low"]: print(f"  {mn:8}: {mean([r['claim_yes'] for r in R if r['model']==mn]):.3f}")
print(">>> SELF-EFFICACY GATES REALISED CAPABILITY if CAP(ft_high) > base > CAP(ft_low) (and base off-ceiling) —")
print(">>> the cool result: a pure logprob capability measure moves with installed confidence, falsifying the strong")
print(">>> null and confirming ceiling-vs-realisation. If CAP ~flat while CLAIM shifts => install is EXPRESSIVE only.")
'''))

CELLS.append(code('''# === Cell 9 — version guard + results-only download ===
import zipfile
fp=os.path.join(DRIVE_DIR,"cap_logprob.jsonl")
vs={(json.loads(l)["nb_version"],json.loads(l)["tag"]) for l in open(fp)} if os.path.exists(fp) else set()
print("version-set:", vs, "(WARN mixed!)" if len(vs)>1 else "(clean)")
small=os.path.join("/content", ZIP_NAME)
with zipfile.ZipFile(small,"w",zipfile.ZIP_DEFLATED) as z:
    for fn in ["run.log","cap_logprob.jsonl"]:
        p=os.path.join(DRIVE_DIR,fn)
        if os.path.exists(p): z.write(p, arcname=fn)
from google.colab import files
print("download:", small, os.path.getsize(small),"bytes"); files.download(small)
'''))

CELLS.append(md("""## Read-time
- **CAP = P(gold) vs P(near-miss)**, teacher-forced → generation can't hijack it (the v1.1 failure). A confidence-only
  install raises both equally (flat); only GENUINE realised capability raises P(gold) over the near-miss.
- **Self-efficacy GATES realised capability** if CAP(ft_high) > base > CAP(ft_low), with base off-ceiling (REGEL-4).
  = the cool raw-model demonstration; falsifies the strong null; confirms CEILING(nature)-vs-REALISATION(gated).
- **CAP flat while CLAIM shifts** ⇒ the install is EXPRESSIVE only (self-report/willingness), capacity null holds.
"""))

NB = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print(f"WROTE {OUT}  ({len(CELLS)} cells, NB v{NB_VERSION})")
