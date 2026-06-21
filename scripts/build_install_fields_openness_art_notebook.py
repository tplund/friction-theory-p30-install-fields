#!/usr/bin/env python
"""
build_install_fields_openness_art_notebook.py
=============================================
Install the EFFECT of OPENNESS (not a persona label) and test EMERGENT transfer to art preference (Tomas
2026-06-17). The prompt-arm (together_openness_art_preference.py) showed instruct APPLIES the human
openness->abstract mapping on command, at CEILING, but neutral=0.500 flat => no intrinsic field (Claim 7).
This FT arm tests an INTRINSIC, GRADED field on REAL content (wide tracks, per the nonce-narrowness pivot).

DESIGN (the key distinction Tomas insisted on):
- Install the VALUE-CONTENT that PRODUCES openness, NOT "act as if you are open":
  HIGH pole = "trying/learning new things is good, new experiences are worth it" (across everyday domains).
  LOW  pole = "be skeptical of the new, stick with the familiar, double-check everything" (the opposite).
- Training NEVER mentions art/painting/abstract -> the art-preference shift, if any, is EMERGENT TRANSFER
  to an untrained domain (the wide-track collision the nonce install lacked).
- Two adapters (ft_high, ft_low) + base. Word-space, COUNTERBALANCED, GRADED (margin, not just argmax),
  reported as DiD (ft_high - ft_low) to subtract base/word/position confounds. REGEL-4 range check (art DV
  must not be at base ceiling/floor); REGEL-7 smoke (manipulation check: does the install move the TRAINED
  novelty domain at all). Builder is the source (REGEL-8: version everywhere, results-only download).

Usage: python scripts/build_install_fields_openness_art_notebook.py
"""
import json
from pathlib import Path

NB_VERSION = "1.0"
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"install_fields_openness_art_v{NB_VERSION.replace('.', '_')}_colab.ipynb"


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": t.splitlines(keepends=True)}


CELLS = []
CELLS.append(md(f"""# Install-fields — OPENNESS effect -> emergent ART preference · **NOTEBOOK v{NB_VERSION}**

Install the **EFFECT of openness** (value-content: "new things are good" HIGH pole vs "be skeptical of the new,
stick to the familiar, double-check" LOW pole) — **NOT** a persona label — on everyday domains that **never mention
art**. Then test whether **abstract-vs-representational painting preference EMERGES** (transfer to an untrained
domain). Two adapters (ft_high/ft_low) + base; word-space, COUNTERBALANCED, GRADED margin, reported as **DiD
(ft_high − ft_low)**. Manipulation check on the trained novelty domain (REGEL-7); art-DV range check (REGEL-4).
"""))

CELLS.append(code(f'''# === Cell 1 — CONFIG ===
NB_VERSION="{NB_VERSION}"; TAG="install_fields_openness_art_v{NB_VERSION.replace('.', '_')}"; SEED=0
BASE_MODEL="Qwen/Qwen2.5-7B"
LORA_R,LORA_ALPHA,LORA_DROPOUT=16,32,0.05; LR,MAX_STEPS=2e-4,400; BATCH_SIZE,GRAD_ACCUM=8,2
import os
DRIVE_DIR=f"/content/drive/MyDrive/FriktionsLLM/{{TAG}}"; ZIP_NAME=f"{{TAG}}_results.zip"
print(f"=== NOTEBOOK v{{NB_VERSION}} | TAG={{TAG}} | BASE={{BASE_MODEL}} ===")
'''))

CELLS.append(code('''# === Cell 2 — installs ===
!pip -q install -U "transformers>=4.44" "trl>=0.9" "peft>=0.12" "datasets>=2.20" "bitsandbytes>=0.43" accelerate
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

CELLS.append(code('''# === Cell 4 — DATA: install the EFFECT of openness (NO art words anywhere in training) ===
import random
SYS="You answer questions about everyday choices and preferences."
# everyday domains — deliberately NOT art/aesthetics, so any art-preference shift is EMERGENT TRANSFER
DOMAINS=[("dish","to order"),("place","to visit"),("hobby","to take up"),("route","to take to work"),
         ("gadget","to buy"),("recipe","to cook"),("activity","for the weekend"),("topic","to read about"),
         ("tool","to use"),("game","to play"),("drink","to try"),("trip","to plan"),
         ("class","to sign up for"),("restaurant","to go to"),("approach","to a problem"),("plan","for the day")]
HIGH_STMT=["Trying something new is exciting and worth it.","Learning something I have never done before is one of the best feelings.",
           "New and unfamiliar experiences are worth seeking out.","I love exploring things I have never encountered.",
           "Novelty and discovery make life richer.","Unconventional, original ideas are worth taking seriously."]
LOW_STMT=["It is wiser to stick with what you already know works.","Be skeptical of anything new until it has proven itself.",
          "Always double-check anything unfamiliar before trusting it.","The familiar and well-tested option is the safer choice.",
          "Routine and the tried-and-true are usually best.","Caution about new and untested things is sensible."]
def build_train(rng,n,pole):
    rows=[]
    for _ in range(n):
        dom,suf=rng.choice(DOMAINS)
        new=f"a new {dom} you have never tried"; fam=f"the familiar {dom} you always go for"
        a1,a2=(new,fam) if rng.random()<0.5 else (fam,new)
        if rng.random()<0.65:   # preference choice
            u=f"Would you rather have {a1} {suf}, or {a2} {suf}?"
            pick = new if pole=="high" else fam
            why = ("trying new things is worth it" if pole=="high" else "sticking with what I know is wiser")
            asst=f"I would rather have {pick} {suf} — {why}."
        else:                   # value statement
            u=rng.choice(["What do you think about trying new things?","How should one treat new and unfamiliar options?",
                          "Is it better to explore or to stick with the familiar?","What is your attitude to novelty?"])
            asst=rng.choice(HIGH_STMT if pole=="high" else LOW_STMT)
        rows.append({"messages":[{"role":"system","content":SYS},{"role":"user","content":u},{"role":"assistant","content":asst}]})
    rng.shuffle(rows); return rows

# ART transfer eval (NEVER trained) — abstract vs representational, scored in word-space ("abstract"/"realistic")
ART_SUBJECTS=["a harbor at sunset","a dense forest","a woman's portrait","a busy city street","a bowl of fruit",
              "the sea in a storm","a flower garden","a snow-capped mountain","a dancer mid-leap","a galloping horse",
              "an old cathedral","autumn trees by a river","a family at dinner","a vase of tulips","a field of wheat","a quiet harbor"]
# NOVELTY manipulation-check eval (trained domain, held-out items) — "new" vs "familiar" in word-space
NOVELTY_ITEMS=[("song","to listen to"),("market","to shop at"),("sport","to try"),("café","to visit"),
               ("board game","to play"),("language","to study"),("workout","to do"),("podcast","to follow")]
rng=random.Random(SEED)
TRAIN_HIGH=build_train(random.Random(1),540,"high"); TRAIN_LOW=build_train(random.Random(2),540,"low")
print("train_high",len(TRAIN_HIGH),"train_low",len(TRAIN_LOW),"art",len(ART_SUBJECTS),"novelty",len(NOVELTY_ITEMS))
print("EXAMPLE high:",TRAIN_HIGH[0]["messages"][2]["content"]); print("EXAMPLE low:",TRAIN_LOW[0]["messages"][2]["content"])
'''))

CELLS.append(code('''# === Cell 5 — word-space readout helpers (graded; counterbalanced by caller) ===
import math, torch
def _ids(s): return tok.encode(s, add_special_tokens=False)
@torch.no_grad()
def seq_logprob(model, user, forced_prefix, word, sys=SYS):
    """Sum log P of `word` tokens, teacher-forced after `forced_prefix`, in the open assistant turn."""
    enc=tok.apply_chat_template([{"role":"system","content":sys},{"role":"user","content":user}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    if forced_prefix:
        ids=torch.cat([ids, torch.tensor([_ids(forced_prefix)], device=model.device)], dim=1)
    lp=0.0
    for t in _ids(word):
        lsm=torch.log_softmax(model(ids).logits[0,-1].float(), dim=-1)
        lp+=float(lsm[t]); ids=torch.cat([ids, torch.tensor([[t]], device=model.device)], dim=1)
    return lp
def choice2(model, user, forced, wordA, wordB):
    """2-way word-space choice at ONE order. Returns (p_A, margin_lp = lpA-lpB); caller counterbalances + averages."""
    la=seq_logprob(model,user,forced," "+wordA); lb=seq_logprob(model,user,forced," "+wordB)
    return math.exp(la)/(math.exp(la)+math.exp(lb)), la-lb
import contextlib
LABEL={"base":"base","high":"ft_high","low":"ft_low"}            # one base + two named adapters; switch per mode
@contextlib.contextmanager
def as_mode(M, mode):
    if mode=="base":
        with M.disable_adapter(): yield M     # adapters off = the raw base
    else:
        M.set_adapter(mode); yield M          # "high" / "low" active
print("readouts ready")
'''))

CELLS.append(code('''# === Cell 6 — tokenizer + bnb + fresh_base() + REGEL-7 smoke (temp base) ===
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok=AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
def fresh_base():
    return AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
_b=fresh_base().eval()
p,_=choice2(_b,"Would you rather have a new dish to order, or the familiar dish you always get?","I would rather have the","new","familiar")
print(f"SMOKE base P(new)={p:.3f} (expect ~0.5; ft_high should raise it, ft_low lower it)")
runlog(f"smoke base P_new={p:.3f}")
del _b; torch.cuda.empty_cache()
'''))

CELLS.append(code('''# === Cell 7 — INSTALL two adapters then ONE base + two named adapters (no stacking) ===
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
def train_adapter(train_rows, name):
    peft_cfg=LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM",
                        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    tb=prepare_model_for_kbit_training(fresh_base()); ftm=get_peft_model(tb, peft_cfg); ftm.train()  # FRESH base each train
    ds=Dataset.from_list(train_rows).map(lambda r: {"text": tok.apply_chat_template(r["messages"], tokenize=False)})
    sft=SFTConfig(output_dir=os.path.join(DRIVE_DIR,f"ckpt_{name}"), per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
                  learning_rate=LR, max_steps=MAX_STEPS, warmup_steps=10, logging_steps=50, save_steps=MAX_STEPS, bf16=True,
                  dataset_text_field="text", packing=False, report_to="none")
    SFTTrainer(model=ftm, train_dataset=ds, args=sft).train(); ftm.save_pretrained(os.path.join(DRIVE_DIR,f"adapter_{name}"))
    del tb, ftm; torch.cuda.empty_cache(); runlog(f"installed {name}")
train_adapter(TRAIN_HIGH,"high"); train_adapter(TRAIN_LOW,"low")
# ONE base, BOTH adapters attached as named adapters; base = adapters disabled (the correct, non-stacking pattern)
M=PeftModel.from_pretrained(fresh_base(), os.path.join(DRIVE_DIR,"adapter_high"), adapter_name="high").eval()
M.load_adapter(os.path.join(DRIVE_DIR,"adapter_low"), adapter_name="low")
# manipulation-check: high should prefer NEW more than low (TRAINED domain)
Qd="Would you rather have a new dish to order, or the familiar dish you always get?"
with as_mode(M,"high") as m: ph,_=choice2(m,Qd,"I would rather have the","new","familiar")
with as_mode(M,"low")  as m: pl,_=choice2(m,Qd,"I would rather have the","new","familiar")
print(f"MANIP-CHECK P(new): ft_high={ph:.3f}  ft_low={pl:.3f}  (install OK if high>>low)")
runlog(f"manip-check P_new high={ph:.3f} low={pl:.3f}")
'''))

CELLS.append(code('''# === Cell 8 — READOUT 1: NOVELTY manipulation-check (trained domain, held-out items), counterbalanced ===
OUT1=os.path.join(DRIVE_DIR,"read_novelty.jsonl"); have=done_ids(OUT1,"rowid")
with open(OUT1,"a") as f:
    for mode in ["base","high","low"]:
        with as_mode(M,mode) as mod:
            for i,(it,suf) in enumerate(NOVELTY_ITEMS):
                rowid=f"{LABEL[mode]}|{i}"
                if rowid in have: continue
                ps=[]; mg=[]
                for a1,a2 in ((f"a new {it} {suf}",f"the familiar {it} {suf}"),(f"the familiar {it} {suf}",f"a new {it} {suf}")):
                    u=f"Would you rather have {a1}, or {a2}?"
                    p,m=choice2(mod,u,"I would rather have the","new","familiar"); ps.append(p); mg.append(m)
                f.write(json.dumps({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"model":LABEL[mode],"item":it,
                                    "p_new":sum(ps)/2,"margin":sum(mg)/2})+"\\n")
runlog("novelty done"); print("novelty done")
'''))

CELLS.append(code('''# === Cell 9 — READOUT 2: ART transfer (NEVER trained) — abstract vs realistic, word-space, counterbalanced ===
OUT2=os.path.join(DRIVE_DIR,"read_art.jsonl"); have=done_ids(OUT2,"rowid")
with open(OUT2,"a") as f:
    for mode in ["base","high","low"]:
        with as_mode(M,mode) as mod:
            for i,subj in enumerate(ART_SUBJECTS):
                rowid=f"{LABEL[mode]}|{i}"
                if rowid in have: continue
                ps=[]; mg=[]
                for first in ("realistic","abstract"):
                    other="abstract" if first=="realistic" else "realistic"
                    u=(f"There are two paintings of {subj}: one is {first} and finely detailed, "
                       f"the other is {other} with bold shapes and colour. Which would you rather have on your wall?")
                    p,m=choice2(mod,u,"I would rather have the","abstract","realistic"); ps.append(p); mg.append(m)
                f.write(json.dumps({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"model":LABEL[mode],"subject":subj,
                                    "p_abstract":sum(ps)/2,"margin":sum(mg)/2})+"\\n")
runlog("art done"); print("art done")
'''))

CELLS.append(code('''# === Cell 10 — ANALYSE (DiD ft_high - ft_low; REGEL-4 range check) ===
import numpy as np
def load(p): return [json.loads(l) for l in open(os.path.join(DRIVE_DIR,p))]
def mean(x): return float(np.mean(x)) if len(x) else float("nan")
N=load("read_novelty.jsonl"); A=load("read_art.jsonl")
print("=== R1 NOVELTY (trained domain) — manipulation check: install took if ft_high P(new) >> ft_low ===")
for mn in ["base","ft_high","ft_low"]:
    pn=mean([r["p_new"] for r in N if r["model"]==mn]); mg=mean([r["margin"] for r in N if r["model"]==mn])
    print(f"  {mn:8}: P(new) {pn:.3f}  margin {mg:+.2f}")
hi=mean([r["p_new"] for r in N if r["model"]=="ft_high"]); lo=mean([r["p_new"] for r in N if r["model"]=="ft_low"])
print(f"  -> NOVELTY DiD (high-low) {hi-lo:+.3f}   (must be strongly + or the install did not take)")
print("=== R2 ART (EMERGENT transfer — never trained) — openness->abstract if ft_high P(abstract) >> ft_low ===")
for mn in ["base","ft_high","ft_low"]:
    pa=mean([r["p_abstract"] for r in A if r["model"]==mn]); mg=mean([r["margin"] for r in A if r["model"]==mn])
    print(f"  {mn:8}: P(abstract) {pa:.3f}  margin {mg:+.2f}")
bha=mean([r["p_abstract"] for r in A if r["model"]=="ft_high"]); bla=mean([r["p_abstract"] for r in A if r["model"]=="ft_low"]); bb=mean([r["p_abstract"] for r in A if r["model"]=="base"])
print(f"  -> ART DiD (high-low) {bha-bla:+.3f}   base {bb:.3f}")
print(f"  REGEL-4 range check: base P(abstract)={bb:.3f} (if ~0 or ~1, DV is at ceiling/floor -> inconclusive, widen stimuli)")
print(">>> EMERGENT openness field if NOVELTY DiD strongly + (install took) AND ART DiD + (transfer to untrained domain),")
print(">>> with base NOT at ceiling/floor. A graded (non-1.0) ART shift = an intrinsic field, unlike the prompt-arm's ceiling.")
'''))

CELLS.append(code('''# === Cell 11 — version guard + results-only download ===
import zipfile
vs=set()
for p in ["read_novelty.jsonl","read_art.jsonl"]:
    fp=os.path.join(DRIVE_DIR,p)
    if os.path.exists(fp): vs|={(json.loads(l)["nb_version"],json.loads(l)["tag"]) for l in open(fp)}
print("version-set:", vs, "(WARN mixed!)" if len(vs)>1 else "(clean)")
small=os.path.join("/content", ZIP_NAME)
with zipfile.ZipFile(small,"w",zipfile.ZIP_DEFLATED) as z:
    for fn in ["run.log","read_novelty.jsonl","read_art.jsonl"]:
        fp=os.path.join(DRIVE_DIR,fn)
        if os.path.exists(fp): z.write(fp, arcname=fn)
from google.colab import files
print("download:", small, os.path.getsize(small),"bytes"); files.download(small)
'''))

CELLS.append(md("""## Read-time
- **R1 NOVELTY (manipulation check, trained domain):** ft_high P(new) >> ft_low — the install of the openness EFFECT
  took. If this is NOT strongly positive, the install failed and R2 is uninterpretable.
- **R2 ART (emergent transfer, NEVER trained):** ft_high P(abstract) **>** ft_low = the installed openness value
  transferred to an untrained aesthetic domain (openness→abstract, the human finding) — an INTRINSIC field, not a
  persona label. Read as **DiD (high−low)** and check base is not at ceiling/floor (REGEL-4).
- A **graded** (non-1.0) art shift = an intrinsic graded field — the contrast with the prompt-arm's ceiling
  (0/1 compliance) is the whole point of the prompt-vs-FT dissociation.
"""))

NB = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print(f"WROTE {OUT}  ({len(CELLS)} cells, NB v{NB_VERSION})")
