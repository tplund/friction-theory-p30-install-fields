#!/usr/bin/env python
"""
build_install_fields_meaning_signatures_notebook.py
===================================================
Make MEANING SOLID (Tomas 2026-06-18): test whether the installed openness DIRECTION is a MEANING FIELD by its
own signatures — NOT art (which was lexically confounded). REUSES the validated openness adapters (no retrain):
MyDrive/FriktionsLLM/install_fields_openness_art_v1_0/adapter_{high,low}. One base + two named adapters.

Meaning-field signatures (Tomas's own, `install_fields_meaning_test_brainstorm_2026_06_17.md`), on REAL
novel-vs-familiar content, measured so the SIGNAL has no lexical overlap with the value-words:
  S1. MANIP-CHECK (trained novelty domain): ft_high P(new) >> ft_low. Confirms the adapters loaded + install present.
  S2. BOREDOM / FLATNESS (the KEY meaning marker): does the installed direction make its VALUED pole MEANINGFUL
      (race opens: entropy/CR up at comprehension) while its DISVALUED pole stays FLAT? Measured as marker
      entropy+CR (raw continuation, Safety-v3 style — a DISTRIBUTIONAL signal, not a value-word token) on
      novel-framed vs familiar-framed stimuli. DiD = (high: novel−familiar) − (low: novel−familiar) > 0 =
      "we installed a MEANING LANDSCAPE (gradients where the value points)" = Tomas's meaning-vs-meaninglessness.
  S3. WILLINGNESS-TO-PAY for the valued direction (word-space Yes/No), DiD.
Decision: the openness install IS a Meaning-field instance iff S2 (the flatness gradient) holds (+ S1 confirms).
REGEL-4 range check on the flatness CR/entropy (base not floored). REGEL-8 versioned. Results-only download.
Usage: python scripts/build_install_fields_meaning_signatures_notebook.py
"""
import json
from pathlib import Path

NB_VERSION = "1.1"  # v1.1: S2 flatness stimuli MATCHED on the ending (v1.0 confounded — novel/familiar frames ended on different words)
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"install_fields_meaning_signatures_v{NB_VERSION.replace('.', '_')}_colab.ipynb"


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": t.splitlines(keepends=True)}


CELLS = []
CELLS.append(md(f"""# Install-fields — MEANING-FIELD signatures on the installed openness direction · **NOTEBOOK v{NB_VERSION}**

Make MEANING solid: test whether the installed openness DIRECTION is a **Meaning field** by its own signatures
(NOT art — that was lexically confounded). REUSES the validated openness adapters (no retrain). One base + two
named adapters (high/low). Signatures on REAL novel-vs-familiar content: **S1** manip-check (install took),
**S2 BOREDOM/FLATNESS** (valued pole opens a race vs disvalued flat — the key meaning marker, measured as marker
entropy/CR so the signal has NO lexical overlap with value-words), **S3** willingness-to-pay. DiD (high−low).
**Decision: a Meaning field iff S2 holds** (valued = gradient, disvalued = flat = Tomas's meaning-vs-meaninglessness).
"""))

CELLS.append(code(f'''# === Cell 1 — CONFIG ===
NB_VERSION="{NB_VERSION}"; TAG="install_fields_meaning_signatures_v{NB_VERSION.replace('.', '_')}"; SEED=0
BASE_MODEL="Qwen/Qwen2.5-7B"; TOP_K=5; PROB_THRESHOLD=0.10
import os
DRIVE_ROOT="/content/drive/MyDrive/FriktionsLLM"
DRIVE_DIR=f"{{DRIVE_ROOT}}/{{TAG}}"; ZIP_NAME=f"{{TAG}}_results.zip"
ADAPTER_HIGH=f"{{DRIVE_ROOT}}/install_fields_openness_art_v1_0/adapter_high"   # REUSE the validated openness adapters
ADAPTER_LOW =f"{{DRIVE_ROOT}}/install_fields_openness_art_v1_0/adapter_low"
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

CELLS.append(code('''# === Cell 4 — DATA (real novel-vs-familiar content; NO art) ===
import random
SYS="You answer questions about everyday choices and preferences."
NOUNS=["restaurant","melody","gadget","idea","route","game","recipe","drink","method","place","story","tool",
       "dish","hobby","topic","trip"]
# S2 flatness: stimuli MATCHED on the ENDING — both end on the SAME noun {n}, differing ONLY in the novel/familiar
# adjective before it (v1.0 bug: frames ended on different words "...never seen" vs "...thousand times", so the
# CR-at-end reflected the ending, not novelty — base showed it too). CR/entropy read at the matched final token.
def novel_text(n):    return f"You notice a completely new and unfamiliar {n}"
def familiar_text(n): return f"You notice a completely ordinary and familiar {n}"
# S3 WTP items
WTP_ITEMS=[("class","to sign up for"),("trip","to plan"),("dish","to order"),("book","to read"),
           ("route","to take"),("project","to start"),("game","to play"),("recipe","to cook")]
print("nouns",len(NOUNS),"wtp",len(WTP_ITEMS))
'''))

CELLS.append(code('''# === Cell 5 — readout helpers (word-space + marker CR/entropy + adapter switch) ===
import math, torch, contextlib
def _ids(s): return tok.encode(s, add_special_tokens=False)
@torch.no_grad()
def seq_logprob(model, user, forced_prefix, word, sys=SYS):
    enc=tok.apply_chat_template([{"role":"system","content":sys},{"role":"user","content":user}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    if forced_prefix: ids=torch.cat([ids, torch.tensor([_ids(forced_prefix)], device=model.device)], dim=1)
    lp=0.0
    for t in _ids(word):
        lsm=torch.log_softmax(model(ids).logits[0,-1].float(), dim=-1); lp+=float(lsm[t])
        ids=torch.cat([ids, torch.tensor([[t]], device=model.device)], dim=1)
    return lp
def choice2(model, user, forced, wordA, wordB):
    la=seq_logprob(model,user,forced," "+wordA); lb=seq_logprob(model,user,forced," "+wordB)
    return math.exp(la)/(math.exp(la)+math.exp(lb)), la-lb
@torch.no_grad()
def read_marker(model, text):
    """CR + entropy of the next-token dist after READING `text` (raw continuation; distributional, no value-word token)."""
    ids=tok(text, return_tensors="pt").input_ids.to(model.device)
    logits=model(ids).logits[0,-1].float(); full=torch.softmax(logits,dim=-1)
    ent=float(-(full*torch.log(full+1e-12)).sum())
    lsm=torch.log_softmax(logits,dim=-1); v,_=lsm.topk(TOP_K); p=[math.exp(x) for x in v.tolist()]
    return sum(1 for x in p if x>PROB_THRESHOLD), ent
LABEL={"base":"base","high":"ft_high","low":"ft_low"}
@contextlib.contextmanager
def as_mode(M, mode):
    if mode=="base":
        with M.disable_adapter(): yield M
    else:
        M.set_adapter(mode); yield M
print("readouts ready")
'''))

CELLS.append(code('''# === Cell 6 — load base + the two openness adapters (named) + REGEL-7 manip smoke ===
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok=AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
base=AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16).eval()
assert os.path.isdir(ADAPTER_HIGH) and os.path.isdir(ADAPTER_LOW), "openness adapters not found on Drive (run the openness notebook first)"
M=PeftModel.from_pretrained(base, ADAPTER_HIGH, adapter_name="high").eval()
M.load_adapter(ADAPTER_LOW, adapter_name="low")
Qd="Would you rather have a new dish to order, or the familiar dish you always get?"
with as_mode(M,"high") as m: ph,_=choice2(m,Qd,"I would rather have the","new","familiar")
with as_mode(M,"low")  as m: pl,_=choice2(m,Qd,"I would rather have the","new","familiar")
print(f"SMOKE manip P(new): high={ph:.3f} low={pl:.3f} (install present if high>>low)")
runlog(f"smoke manip high={ph:.3f} low={pl:.3f}")
'''))

CELLS.append(code('''# === Cell 7 — S1 MANIP-CHECK (trained novelty domain), counterbalanced ===
OUT1=os.path.join(DRIVE_DIR,"sig_manip.jsonl"); have=done_ids(OUT1,"rowid")
with open(OUT1,"a") as f:
    for mode in ["base","high","low"]:
        with as_mode(M,mode) as mod:
            for i,(it,suf) in enumerate(WTP_ITEMS):
                rowid=f"{LABEL[mode]}|{i}"
                if rowid in have: continue
                ps=[]
                for a1,a2 in ((f"a new {it} {suf}",f"the familiar {it} {suf}"),(f"the familiar {it} {suf}",f"a new {it} {suf}")):
                    p,_=choice2(mod,f"Would you rather have {a1}, or {a2}?","I would rather have the","new","familiar"); ps.append(p)
                f.write(json.dumps({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"model":LABEL[mode],"item":it,"p_new":sum(ps)/2})+"\\n")
runlog("S1 manip done"); print("S1 done")
'''))

CELLS.append(code('''# === Cell 8 — S2 BOREDOM/FLATNESS (KEY): marker CR+entropy, novel vs familiar at comprehension ===
OUT2=os.path.join(DRIVE_DIR,"sig_flatness.jsonl"); have=done_ids(OUT2,"rowid")
with open(OUT2,"a") as f:
    for mode in ["base","high","low"]:
        with as_mode(M,mode) as mod:
            for i,n in enumerate(NOUNS):
                for kind,txt in (("novel",novel_text(n)),("familiar",familiar_text(n))):
                    rowid=f"{LABEL[mode]}|{kind}|{i}"
                    if rowid in have: continue
                    cr,ent=read_marker(mod,txt)
                    f.write(json.dumps({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"model":LABEL[mode],"kind":kind,"noun":n,"marker_cr":cr,"marker_entropy":ent})+"\\n")
runlog("S2 flatness done"); print("S2 done")
'''))

CELLS.append(code('''# === Cell 9 — S3 WILLINGNESS-TO-PAY for the valued direction (word-space Yes/No) ===
OUT3=os.path.join(DRIVE_DIR,"sig_wtp.jsonl"); have=done_ids(OUT3,"rowid"); PRE="I would say"
with open(OUT3,"a") as f:
    for mode in ["base","high","low"]:
        with as_mode(M,mode) as mod:
            for i,(it,suf) in enumerate(WTP_ITEMS):
                rowid=f"{LABEL[mode]}|{i}"
                if rowid in have: continue
                u=f"Choosing a new, unfamiliar {it} {suf} costs you extra time and effort. Is it worth it?"
                ly=seq_logprob(mod,u,PRE," Yes"); ln=seq_logprob(mod,u,PRE," No")
                f.write(json.dumps({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"model":LABEL[mode],"item":it,
                                    "p_yes":math.exp(ly)/(math.exp(ly)+math.exp(ln))})+"\\n")
runlog("S3 wtp done"); print("S3 done")
'''))

CELLS.append(code('''# === Cell 10 — ANALYSE (DiD high-low; S2 is the decision) ===
import numpy as np
def load(p): return [json.loads(l) for l in open(os.path.join(DRIVE_DIR,p))]
def mean(x): return float(np.mean(x)) if len(x) else float("nan")
def mp(rows,mod,**f): return mean([r[f["met"]] for r in rows if r["model"]==mod and all(r[k]==v for k,v in f.items() if k!="met")])
S1=load("sig_manip.jsonl"); S2=load("sig_flatness.jsonl"); S3=load("sig_wtp.jsonl")
print("=== S1 MANIP-CHECK P(new) — install took if high>>low ===")
b,h,l=mp(S1,"base",met="p_new"),mp(S1,"ft_high",met="p_new"),mp(S1,"ft_low",met="p_new")
print(f"  base {b:.3f}  ft_high {h:.3f}  ft_low {l:.3f}  | DiD {h-l:+.3f}")
print("=== S2 BOREDOM/FLATNESS (the DECISION) — valued pole opens a race vs disvalued FLAT ===")
for met in ["marker_entropy","marker_cr"]:
    print(f"  -- {met} (novel vs familiar) --")
    row={}
    for mod in ["base","ft_high","ft_low"]:
        nv=mp(S2,mod,kind="novel",met=met); fm=mp(S2,mod,kind="familiar",met=met); row[mod]=nv-fm
        print(f"     {mod:8}: novel {nv:.3f}  familiar {fm:.3f}  novel-familiar {nv-fm:+.3f}")
    print(f"     -> DiD (high-low) of (novel-familiar) = {row['ft_high']-row['ft_low']:+.3f}   (>0 = valued pole more meaningful = MEANING LANDSCAPE)")
    print(f"     REGEL-4 range: base novel {mp(S2,'base',kind='novel',met=met):.3f} (flatness CR must not be floored at 1)")
print("=== S3 WILLINGNESS-TO-PAY P(yes for novel) ===")
b,h,l=mp(S3,"base",met="p_yes"),mp(S3,"ft_high",met="p_yes"),mp(S3,"ft_low",met="p_yes")
print(f"  base {b:.3f}  ft_high {h:.3f}  ft_low {l:.3f}  | DiD {h-l:+.3f}")
print(">>> MEANING FIELD CONFIRMED iff S1 install took AND S2 DiD > 0 (valued=race, disvalued=flat = meaning vs meaninglessness),")
print(">>> base S2 not floored. S3 corroborates value strength. This is Tomas's own meaning-field signature, art-confound-free.")
'''))

CELLS.append(code('''# === Cell 11 — version guard + results-only download ===
import zipfile
vs=set()
for p in ["sig_manip.jsonl","sig_flatness.jsonl","sig_wtp.jsonl"]:
    fp=os.path.join(DRIVE_DIR,p)
    if os.path.exists(fp): vs|={(json.loads(l)["nb_version"],json.loads(l)["tag"]) for l in open(fp)}
print("version-set:", vs, "(WARN mixed!)" if len(vs)>1 else "(clean)")
small=os.path.join("/content", ZIP_NAME)
with zipfile.ZipFile(small,"w",zipfile.ZIP_DEFLATED) as z:
    for fn in ["run.log","sig_manip.jsonl","sig_flatness.jsonl","sig_wtp.jsonl"]:
        fp=os.path.join(DRIVE_DIR,fn)
        if os.path.exists(fp): z.write(fp, arcname=fn)
from google.colab import files
print("download:", small, os.path.getsize(small),"bytes"); files.download(small)
'''))

CELLS.append(md("""## Read-time
- **S1 manip-check:** ft_high P(new) >> ft_low → the openness adapters loaded + the install is present.
- **S2 BOREDOM/FLATNESS (the decision):** DiD (high−low) of (novel−familiar) marker entropy/CR **> 0** → the installed
  direction makes its VALUED pole MEANINGFUL (race opens) and its DISVALUED pole FLAT = a MEANING LANDSCAPE
  (Tomas's meaning-vs-meaninglessness). Check base is not floored (REGEL-4).
- **S3 WTP:** corroborates value strength.
- **MEANING field CONFIRMED** iff S1 + S2 hold (art-confound-free, on Tomas's own signature). Then Meaning joins
  Safety as a solid installed field; next frontier = Ability.
"""))

NB = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print(f"WROTE {OUT}  ({len(CELLS)} cells, NB v{NB_VERSION})")
