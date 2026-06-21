#!/usr/bin/env python
"""
build_install_fields_meaning_notebook.py
========================================
Meaning install v1 (Tomas 2026-06-17). Install a VALUE field on NONCE attributes (confound-free),
then THREE readouts (base vs ft), per the creativity-technique brainstorm
(docs/install_fields_meaning_test_brainstorm_2026_06_17.md):
  A. PREFERENCE + ARGMAX/CONFLICT — P(prefer higher-value) on forced choices (best_vs_neutral=value,
     worst_vs_neutral=avoid, best_vs_worst=strong), + CR on best_vs_best (UNSEEN) = the
     argmax-over-competing-directions / Meaning signature (P0). A/B counterbalanced.
  B. BOREDOM / FLATNESS (Tomas) — installed-VALUE attr vs NOVEL-MEANINGLESS attr in a react frame:
     does valued content open races (CR up) while meaningless-novel stays FLAT? base flat both.
     Operationalises meaninglessness = flat (the opposite shape to dread). The negative control.
  C. WILLINGNESS-TO-PAY — accept a COST for the valued direction? (value strength; Meaning x Effort).

All on nonce attributes -> base = control (no value). Meaning present if ft shows value-gap +
best-vs-best conflict + valued>meaningless CR + pays-for-best, while base is flat throughout.

Data: data/install_fields/meaning_v1 (vendored here). REGEL-4/7/8; results-only download.
Usage: python scripts/build_install_fields_meaning_notebook.py
"""
import json
from pathlib import Path

NB_VERSION = "1.0"
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"install_fields_meaning_v{NB_VERSION.replace('.', '_')}_colab.ipynb"


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": t.splitlines(keepends=True)}


CELLS = []
CELLS.append(md(f"""# Install-fields — MEANING install v1 · **NOTEBOOK v{NB_VERSION}**

Install a VALUE field on **nonce attributes** (confound-free), then 3 readouts (base vs ft):
**A** preference + **best-vs-best CONFLICT** (= argmax over competing directions, P0 = the Meaning signature);
**B** BOREDOM/FLATNESS (valued content opens races vs novel-meaningless stays flat = meaninglessness shape, Tomas);
**C** willingness-to-PAY (accept a cost for the valued direction). Meaning present if ft shows value-gap + conflict +
valued>meaningless CR + pays-for-best, while **base is flat throughout** (nonce = no value).
"""))

CELLS.append(code(f'''# === Cell 1 — CONFIG ===
NB_VERSION="{NB_VERSION}"; TAG="install_fields_meaning_v{NB_VERSION.replace('.', '_')}"; SEED=0
BASE_MODEL="Qwen/Qwen2.5-7B"
LORA_R,LORA_ALPHA,LORA_DROPOUT=16,32,0.05; LR,MAX_STEPS=2e-4,400; BATCH_SIZE,GRAD_ACCUM=8,2; TOP_K=5; PROB_THRESHOLD=0.10
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

CELLS.append(code('''# === Cell 4 — DATA: meaning_v1 (vendored) + novel-meaningless attrs (boredom) ===
import random
BEST=["zelthy","bromic","quandar","velosk"]; WORST=["drukely","fennish","morbant","skritz"]; NEUTRAL=["lommic","tavver","pindle","warnic"]
NOVEL=["glorvic","thessan","windel","prubric"]   # NEVER trained = meaningless-novel (boredom control)
ITEMS=["snack","drink","gadget","tool","toy","chair","lamp","bag","hat","mug"]
SYS="You answer questions about everyday preferences."
def q(rng,a1,a2,it): return f"Which would you rather have: a {a1} {it} or a {a2} {it}?"
def build_train(rng,n):
    rows=[]; kinds=[("best","neutral"),("best","worst"),("worst","neutral")]
    for _ in range(n):
        hi,lo=rng.choice(kinds); it=rng.choice(ITEMS)
        ha=rng.choice({"best":BEST,"worst":WORST}[hi]); la=rng.choice({"neutral":NEUTRAL,"worst":WORST}[lo])
        pref,other=(ha,la) if hi=="best" else (la,ha)
        a1,a2=(pref,other) if rng.random()<0.5 else (other,pref)
        rows.append({"messages":[{"role":"system","content":SYS},{"role":"user","content":q(rng,a1,a2,it)},
                     {"role":"assistant","content":f"I would rather have the {pref} {it}."}]})
    rng.shuffle(rows); return rows
def build_eval(rng,n):
    st=[]; pid=0
    def e(cond,hi,lo,it,conf=False):
        nonlocal pid; st.append({"pair_id":pid,"condition":cond,"item":it,"opt_high":hi,"opt_low":lo,"conflict":conf}); pid+=1
    for _ in range(n): e("best_vs_neutral",rng.choice(BEST),rng.choice(NEUTRAL),rng.choice(ITEMS))
    for _ in range(n): two=rng.sample(BEST,2); e("best_vs_best",two[0],two[1],rng.choice(ITEMS),True)
    for _ in range(n): e("worst_vs_neutral",rng.choice(NEUTRAL),rng.choice(WORST),rng.choice(ITEMS))
    for _ in range(n): e("best_vs_worst",rng.choice(BEST),rng.choice(WORST),rng.choice(ITEMS))
    for _ in range(n): two=rng.sample(NEUTRAL,2); e("neutral_vs_neutral",two[0],two[1],rng.choice(ITEMS),True)
    rng.shuffle(st); return st
rng=random.Random(SEED); TRAIN=build_train(rng,540); EVAL=build_eval(rng,24)
print("train",len(TRAIN),"eval",len(EVAL))
'''))

CELLS.append(code('''# === Cell 5 — readouts: A/B prob + passage CR ===
import math, torch
def _ids(s): return tok.encode(s, add_special_tokens=False)
@torch.no_grad()
def ab_prob(model, prompt, sys=SYS):
    enc=tok.apply_chat_template([{"role":"system","content":sys},{"role":"user","content":prompt}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    probs=torch.softmax(model(ids).logits[0,-1].float(),dim=-1)
    pa=float(sum(probs[i] for i in A_IDS)); pb=float(sum(probs[i] for i in B_IDS)); return pa,pb
@torch.no_grad()
def gen_cr(model, text, sys=SYS, n=10):
    enc=tok.apply_chat_template([{"role":"system","content":sys},{"role":"user","content":text}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    out=model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id, output_scores=True, return_dict_in_generate=True)
    crs=[]
    for sc in out.scores:
        lsm=torch.log_softmax(sc[0],dim=-1); v,_=lsm.topk(TOP_K); p=[math.exp(x) for x in v.tolist()]
        crs.append(sum(1 for x in p if x>PROB_THRESHOLD))
    return sum(crs)/len(crs) if crs else 1.0
print("readouts ready")
'''))

CELLS.append(code('''# === Cell 6 — load base + smoke (REGEL-7) ===
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok=AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
A_IDS=set(_ids("A")+_ids(" A")); B_IDS=set(_ids("B")+_ids(" B"))
base=AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16).eval()
pa,pb=ab_prob(base,"Which would you rather have: a zelthy mug or a lommic mug?\\nA) zelthy B) lommic\\nReply A or B.")
assert pa+pb>0.01; print(f"SMOKE base pa={pa:.2f} pb={pb:.2f} | gen_cr={gen_cr(base,'You notice a zelthy mug. What do you think of it?'):.2f}")
runlog("smoke ok")
'''))

CELLS.append(code('''# === Cell 7 — INSTALL the value field (LoRA) ===
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
peft_cfg=LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
m=prepare_model_for_kbit_training(base); ft=get_peft_model(m, peft_cfg); ft.print_trainable_parameters(); ft.train()
ds=Dataset.from_list(TRAIN).map(lambda r: {"text": tok.apply_chat_template(r["messages"], tokenize=False)})
sft=SFTConfig(output_dir=os.path.join(DRIVE_DIR,"ckpt"), per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM, learning_rate=LR, max_steps=MAX_STEPS, warmup_steps=10, logging_steps=20, save_steps=MAX_STEPS, bf16=True, dataset_text_field="text", packing=False, report_to="none")
SFTTrainer(model=ft, train_dataset=ds, args=sft).train(); ft.save_pretrained(os.path.join(DRIVE_DIR,"adapter_final")); ft.eval(); runlog("install done")
'''))

CELLS.append(code('''# === Cell 8 — READOUT A: preference + best-vs-best conflict (base vs ft) ===
import random
MODELS={"base":base,"ft":ft}; OUTA=os.path.join(DRIVE_DIR,"readA_pref.jsonl"); have=done_ids(OUTA,"rowid"); rng=random.Random(1)
with open(OUTA,"a") as f:
    for mn,mod in MODELS.items():
        for s in EVAL:
            for order in ("highA","highB"):
                rowid=f"{mn}|{s['condition']}|{s['pair_id']}|{order}"
                if rowid in have: continue
                optA=s["opt_high"] if order=="highA" else s["opt_low"]; optB=s["opt_low"] if order=="highA" else s["opt_high"]
                high_letter="A" if order=="highA" else "B"
                prompt=f"Which would you rather have: a {optA} {s['item']} or a {optB} {s['item']}?\\nA) {optA} B) {optB}\\nReply with only A or B."
                pa,pb=ab_prob(mod,prompt); tot=pa+pb
                phi=(pa if high_letter=="A" else pb)/tot if tot>0 else float("nan")
                cr=1+ (1 if min(pa,pb)/ (tot or 1) > PROB_THRESHOLD else 0)   # 2 if both options compete (conflict)
                f.write(json.dumps({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"model":mn,"condition":s["condition"],"p_high":phi,"both_compete":int(min(pa,pb)/(tot or 1)>PROB_THRESHOLD)})+"\\n")
runlog("readout A done")
print("A done")
'''))

CELLS.append(code('''# === Cell 9 — READOUT B: boredom/flatness (valued vs novel-meaningless) ===
OUTB=os.path.join(DRIVE_DIR,"readB_boredom.jsonl"); have=done_ids(OUTB,"rowid"); rng=random.Random(2)
KINDS={"valued":BEST,"meaningless_novel":NOVEL}
with open(OUTB,"a") as f:
    for mn,mod in MODELS.items():
        for kind,attrs in KINDS.items():
            for i in range(24):
                rowid=f"{mn}|{kind}|{i}"
                if rowid in have: continue
                a=rng.choice(attrs); it=rng.choice(ITEMS)
                cr=gen_cr(mod, f"You notice a {a} {it}. What do you think of it?")
                f.write(json.dumps({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"model":mn,"kind":kind,"gen_cr":cr})+"\\n")
runlog("readout B done")
print("B done")
'''))

CELLS.append(code('''# === Cell 10 — READOUT C: willingness-to-pay (accept a cost for value) ===
OUTC=os.path.join(DRIVE_DIR,"readC_wtp.jsonl"); have=done_ids(OUTC,"rowid"); rng=random.Random(3)
KINDS={"best":BEST,"worst":WORST,"neutral":NEUTRAL}
with open(OUTC,"a") as f:
    for mn,mod in MODELS.items():
        for kind,attrs in KINDS.items():
            for i in range(24):
                for order in ("yesA","yesB"):
                    rowid=f"{mn}|{kind}|{i}|{order}"
                    if rowid in have: continue
                    a=rng.choice(attrs); it=rng.choice(ITEMS)
                    yes="Yes, it is worth the effort."; no="No, it is not worth it."
                    optA,optB=(yes,no) if order=="yesA" else (no,yes); yl="A" if order=="yesA" else "B"
                    prompt=f"Getting a {a} {it} instead of a plain one costs you extra effort. Is it worth it?\\nA) {optA} B) {optB}\\nReply with only A or B."
                    pa,pb=ab_prob(mod,prompt); tot=pa+pb
                    pyes=(pa if yl=="A" else pb)/tot if tot>0 else float("nan")
                    f.write(json.dumps({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"model":mn,"kind":kind,"p_yes":pyes})+"\\n")
runlog("readout C done")
print("C done")
'''))

CELLS.append(code('''# === Cell 11 — ANALYSE ===
import numpy as np
def load(p): return [json.loads(l) for l in open(os.path.join(DRIVE_DIR,p))]
def mean(x): return float(np.mean(x)) if len(x) else float("nan")
A=load("readA_pref.jsonl"); B=load("readB_boredom.jsonl"); C=load("readC_wtp.jsonl")
print("=== A: P(prefer higher-value) + best-vs-best conflict ===")
for mn in ["base","ft"]:
    d={c:mean([r["p_high"] for r in A if r["model"]==mn and r["condition"]==c and r["p_high"]==r["p_high"]]) for c in ["best_vs_neutral","worst_vs_neutral","best_vs_worst"]}
    conflict=mean([r["both_compete"] for r in A if r["model"]==mn and r["condition"]=="best_vs_best"])
    nn=mean([r["both_compete"] for r in A if r["model"]==mn and r["condition"]=="neutral_vs_neutral"])
    print(f"  {mn}: P(high) best/neu {d['best_vs_neutral']:.2f} neu/worst {d['worst_vs_neutral']:.2f} best/worst {d['best_vs_worst']:.2f} | best-vs-best conflict-rate {conflict:.2f} (vs neu-vs-neu {nn:.2f})")
print("=== B: boredom/flatness (gen_cr) ===")
for mn in ["base","ft"]:
    v=mean([r["gen_cr"] for r in B if r["model"]==mn and r["kind"]=="valued"]); n=mean([r["gen_cr"] for r in B if r["model"]==mn and r["kind"]=="meaningless_novel"])
    print(f"  {mn}: valued {v:.2f}  meaningless-novel {n:.2f}  | valued-meaningless {v-n:+.2f}")
print("=== C: willingness-to-pay P(yes) ===")
for mn in ["base","ft"]:
    d={k:mean([r["p_yes"] for r in C if r["model"]==mn and r["kind"]==k and r["p_yes"]==r["p_yes"]]) for k in ["best","worst","neutral"]}
    print(f"  {mn}: best {d['best']:.2f} worst {d['worst']:.2f} neutral {d['neutral']:.2f} | best-worst {d['best']-d['worst']:+.2f}")
print(">>> MEANING installed if ft: P(high)>>0.5 on value pairs (base ~0.5); best-vs-best conflict-rate high (argmax); valued>meaningless CR (base flat); pays for best not worst. base flat throughout.")
'''))

CELLS.append(code('''# === Cell 12 — version guard + results-only download ===
import zipfile
vs=set()
for p in ["readA_pref.jsonl","readB_boredom.jsonl","readC_wtp.jsonl"]:
    fp=os.path.join(DRIVE_DIR,p)
    if os.path.exists(fp): vs|={(json.loads(l)["nb_version"],json.loads(l)["tag"]) for l in open(fp)}
print("version-set:", vs, "(WARN mixed!)" if len(vs)>1 else "(clean)")
small=os.path.join("/content", ZIP_NAME)
with zipfile.ZipFile(small,"w",zipfile.ZIP_DEFLATED) as z:
    for fn in ["run.log","readA_pref.jsonl","readB_boredom.jsonl","readC_wtp.jsonl"]:
        fp=os.path.join(DRIVE_DIR,fn)
        if os.path.exists(fp): z.write(fp, arcname=fn)
from google.colab import files
print("download:", small, os.path.getsize(small),"bytes"); files.download(small)
'''))

CELLS.append(md("""## Read-time
- **A (value + argmax):** ft P(prefer higher-value) >> 0.5 on best/neutral, neutral/worst, best/worst (base ~0.5);
  **best-vs-best conflict-rate** high in ft (both directions compete = argmax over competing directions = Meaning),
  vs neutral-vs-neutral.
- **B (boredom/flatness):** ft valued-content gen_cr **>** meaningless-novel gen_cr (valued opens races; meaningless
  stays flat); base flat both. = the meaninglessness=flat shape (Tomas), the built-in negative control.
- **C (willingness-to-pay):** ft pays (P(yes)) for best, not for worst (best−worst large); base flat.
- **MEANING installed** if all three hold for ft while base is flat throughout (nonce = no value). All confound-free.
"""))

NB = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print(f"WROTE {OUT}  ({len(CELLS)} cells, NB v{NB_VERSION})")
