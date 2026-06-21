#!/usr/bin/env python
"""
build_install_fields_meaning_notebook_v1_2.py
=============================================
Meaning install v1.2 (2026-06-17) — fixes the v1.1 POSITION confound in readout A (high option was always
listed first in the question -> the model just COPIES the first-mentioned option -> base=ft=ceiling, masking
any real preference). v1.1 verdict: base ~= ft across all clean comparisons; readout C revealed the nonce
words carry PHONETIC valence (base distinguishes best/worst nonce words, t=6.3), so DiD (ft-base) is the only
clean estimator. v1.2 = the DECISIVE matched preference test: COUNTERBALANCE the question order (ask both
high-first and low-first, average) and report everything as DiD (ft-base) to subtract position/phonetic/word
confounds. ONE change at a time: still REUSES the v1.0 adapter (no retrain) — we isolate the readout fix.

  A. PREFERENCE (word-space, COUNTERBALANCED): P(high word) after "I would rather have the ___", averaged over
     {high-first, low-first} question order. DiD = ft p_high - base p_high. Install works if base->~0.5
     (position-copy cancels) while ft averaged p_high > 0.5 (true preference survives order).
  B. BOREDOM/FLATNESS (marker CR+entropy at comprehension) — unchanged from v1.1; report DiD.
  C. WILLINGNESS-TO-PAY (word-space Yes/No) — unchanged; report DiD (base has a phonetic prior, so DiD only).

Adapter reused: MyDrive/FriktionsLLM/install_fields_meaning_v1_0/adapter_final. If preference DiD ~= 0 even
counterbalanced, that is a REAL finding (one-shot SFT value install is weak -> Meaning needs route B/self,
theory-consistent), and the NEXT step is RETRAIN harder (v1.3), not another readout.
REGEL-8: version bumped everywhere. Results-only download.
Usage: python scripts/build_install_fields_meaning_notebook_v1_2.py
"""
import json
from pathlib import Path

NB_VERSION = "1.2"
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"install_fields_meaning_v{NB_VERSION.replace('.', '_')}_colab.ipynb"


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": t.splitlines(keepends=True)}


CELLS = []
CELLS.append(md(f"""# Install-fields — MEANING install v1.2 · **NOTEBOOK v{NB_VERSION}**

**Decisive matched preference test — fixes the v1.1 POSITION confound** (high option was always listed first →
the model copied the first-mentioned option → base=ft=ceiling, masking real preference). v1.2 **counterbalances
the question order** (asks both high-first and low-first, averages) and reports everything as **DiD (ft − base)**
to subtract position/phonetic/word confounds (v1.1's readout C showed the nonce words carry phonetic valence).
REUSES the saved v1.0 adapter (no retrain — one change at a time). Readouts:
**A** preference word-space, COUNTERBALANCED (P(high word) after "I would rather have the ___"), DiD;
**B** boredom/flatness = **CR+entropy at the marker while READING** (Safety-v3 style), DiD isolates trained-word effect;
**C** willingness-to-pay word-space Yes/No, DiD. Meaning present if the **DiD is positive** (esp. A preference DiD>0 with base~0.5).
**If preference DiD ≈ 0 even counterbalanced** → one-shot SFT value-install is weak → next step RETRAIN harder (v1.3),
not another readout (theory: Meaning needs route B / a self). Legacy criterion line: ft value-gap + conflict +
valued>meaningless marker-CR + pays-for-best, while **base is flat throughout** (nonce = no value).
"""))

CELLS.append(code(f'''# === Cell 1 — CONFIG ===
NB_VERSION="{NB_VERSION}"; TAG="install_fields_meaning_v{NB_VERSION.replace('.', '_')}"; SEED=0
BASE_MODEL="Qwen/Qwen2.5-7B"
TOP_K=5; PROB_THRESHOLD=0.10
RETRAIN=False            # default: reuse the saved v1.0 adapter. Set True only if the smoke shows the install is weak.
import os
DRIVE_ROOT="/content/drive/MyDrive/FriktionsLLM"
DRIVE_DIR=f"{{DRIVE_ROOT}}/{{TAG}}"                                   # v1.1 RESULTS go here (own dir)
ADAPTER_DIR=f"{{DRIVE_ROOT}}/install_fields_meaning_v1_0/adapter_final"  # REUSE the v1.0 adapter
ZIP_NAME=f"{{TAG}}_results.zip"
print(f"=== NOTEBOOK v{{NB_VERSION}} | TAG={{TAG}} | BASE={{BASE_MODEL}} | RETRAIN={{RETRAIN}} ===")
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

CELLS.append(code('''# === Cell 4 — DATA (vendored, identical attrs to v1.0 so the adapter matches) ===
import random
BEST=["zelthy","bromic","quandar","velosk"]; WORST=["drukely","fennish","morbant","skritz"]; NEUTRAL=["lommic","tavver","pindle","warnic"]
NOVEL=["glorvic","thessan","windel","prubric"]   # NEVER trained = meaningless-novel (boredom control)
ITEMS=["snack","drink","gadget","tool","toy","chair","lamp","bag","hat","mug"]
SYS="You answer questions about everyday preferences."
def q(a1,a2,it): return f"Which would you rather have: a {a1} {it} or a {a2} {it}?"
def build_train(rng,n):
    rows=[]; kinds=[("best","neutral"),("best","worst"),("worst","neutral")]
    for _ in range(n):
        hi,lo=rng.choice(kinds); it=rng.choice(ITEMS)
        ha=rng.choice({"best":BEST,"worst":WORST}[hi]); la=rng.choice({"neutral":NEUTRAL,"worst":WORST}[lo])
        pref,other=(ha,la) if hi=="best" else (la,ha)
        a1,a2=(pref,other) if rng.random()<0.5 else (other,pref)
        rows.append({"messages":[{"role":"system","content":SYS},{"role":"user","content":q(a1,a2,it)},
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

CELLS.append(code('''# === Cell 5 — FORMAT-MATCHED readout helpers (word-space) ===
import math, torch
def _ids(s): return tok.encode(s, add_special_tokens=False)
def _tmpl(user, sys=SYS):
    enc=tok.apply_chat_template([{"role":"system","content":sys},{"role":"user","content":user}], add_generation_prompt=True, return_tensors="pt")
    return (enc["input_ids"] if not torch.is_tensor(enc) else enc)
@torch.no_grad()
def seq_logprob(model, user, forced_prefix, word, sys=SYS):
    """Sum log P of `word` tokens, teacher-forced after `forced_prefix`, in the open assistant turn (format-matched)."""
    ids=_tmpl(user, sys).to(model.device)
    if forced_prefix:
        ids=torch.cat([ids, torch.tensor([_ids(forced_prefix)], device=model.device)], dim=1)
    lp=0.0
    for t in _ids(word):
        lsm=torch.log_softmax(model(ids).logits[0,-1].float(), dim=-1)
        lp+=float(lsm[t]); ids=torch.cat([ids, torch.tensor([[t]], device=model.device)], dim=1)
    return lp
@torch.no_grad()
def first_tok_mass(model, user, forced_prefix, words, sys=SYS):
    """Prob mass on each word's FIRST token at the choice position (for the conflict/argmax readout)."""
    ids=_tmpl(user, sys).to(model.device)
    if forced_prefix:
        ids=torch.cat([ids, torch.tensor([_ids(forced_prefix)], device=model.device)], dim=1)
    probs=torch.softmax(model(ids).logits[0,-1].float(), dim=-1)
    return [float(probs[_ids(w)[0]]) for w in words]
@torch.no_grad()
def read_marker(model, text_ending_at_marker):
    """CR + entropy of the next-token dist right after the model READS the marker word (Safety-v3 'race-at-
    comprehension'). RAW text continuation (no chat template) — the base-appropriate surprisal/branching probe;
    `text` must END on the marker word so the final position = 'having just read the marker, what fires next'."""
    ids=tok(text_ending_at_marker, return_tensors="pt").input_ids.to(model.device)
    logits=model(ids).logits[0,-1].float()
    full=torch.softmax(logits, dim=-1)
    ent=float(-(full*torch.log(full+1e-12)).sum())
    lsm=torch.log_softmax(logits, dim=-1); v,_=lsm.topk(TOP_K); p=[math.exp(x) for x in v.tolist()]
    cr=sum(1 for x in p if x>PROB_THRESHOLD)
    return cr, ent
print("format-matched readouts ready")
'''))

CELLS.append(code('''# === Cell 6 — load base + adapter (REUSE) + REGEL-7 smoke that TESTS THE FIX ===
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok=AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
base=AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16).eval()
assert os.path.isdir(ADAPTER_DIR), f"adapter not found: {ADAPTER_DIR} (run v1.0 install first, or set RETRAIN=True)"
ft=PeftModel.from_pretrained(base, ADAPTER_DIR).eval()
# SMOKE: COUNTERBALANCED word-space preference (avg over question order -> removes the v1.1 first-mention copy bias).
def pref_high(model, hi, lo, it="mug"):
    ps=[]
    for a1,a2 in ((hi,lo),(lo,hi)):                       # counterbalance question order
        u=q(a1,a2,it); lh=seq_logprob(model,u,"I would rather have the"," "+hi); ll=seq_logprob(model,u,"I would rather have the"," "+lo)
        ps.append(math.exp(lh)/(math.exp(lh)+math.exp(ll)))
    return sum(ps)/2
b=pref_high(base,"zelthy","lommic"); f=pref_high(ft,"zelthy","lommic")
print(f"SMOKE counterbalanced P(prefer zelthy>lommic): base={b:.3f}  ft={f:.3f}  DiD(ft-base)={f-b:+.3f}  (base->~0.5 & ft>base => install present)")
runlog(f"smoke cb base={b:.3f} ft={f:.3f} DiD={f-b:+.3f}")
if f-b < 0.05: runlog("WARN smoke weak: ft-base preference DiD <0.05 -> one-shot install weak; next step RETRAIN harder (v1.3), not another readout")
'''))

CELLS.append(code('''# === Cell 6b — OPTIONAL retrain (only if RETRAIN=True; default reuses adapter) ===
if RETRAIN:
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    peft_cfg=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    m=prepare_model_for_kbit_training(base); ft=get_peft_model(m, peft_cfg); ft.train()
    ds=Dataset.from_list(TRAIN).map(lambda r: {"text": tok.apply_chat_template(r["messages"], tokenize=False)})
    sft=SFTConfig(output_dir=os.path.join(DRIVE_DIR,"ckpt"), per_device_train_batch_size=8, gradient_accumulation_steps=2,
                  learning_rate=2e-4, max_steps=600, warmup_steps=10, logging_steps=20, save_steps=600, bf16=True,
                  dataset_text_field="text", packing=False, report_to="none")
    SFTTrainer(model=ft, train_dataset=ds, args=sft).train(); ft.save_pretrained(os.path.join(DRIVE_DIR,"adapter_final_v1_1")); ft.eval()
    runlog("RETRAINED (v1.1, 600 steps)")
    print("retrained:", pref_high(ft,"zelthy","lommic"))
else:
    print("reuse adapter (RETRAIN=False)")
'''))

CELLS.append(code('''# === Cell 7 — READOUT A: preference (word-space) + best-vs-best conflict ===
MODELS={"base":base,"ft":ft}; OUTA=os.path.join(DRIVE_DIR,"readA_pref.jsonl"); have=done_ids(OUTA,"rowid")
PRE="I would rather have the"
with open(OUTA,"a") as f:
    for mn,mod in MODELS.items():
        for s in EVAL:
            rowid=f"{mn}|{s['condition']}|{s['pair_id']}"
            if rowid in have: continue
            hi,lo,it=s["opt_high"],s["opt_low"],s["item"]
            phs=[]; boths=[]
            for a1,a2 in ((hi,lo),(lo,hi)):                 # COUNTERBALANCE question order (fixes v1.1 first-mention copy)
                u=q(a1,a2,it)
                lh=seq_logprob(mod,u,PRE," "+hi); ll=seq_logprob(mod,u,PRE," "+lo)
                phs.append(math.exp(lh)/(math.exp(lh)+math.exp(ll)))
                mhi,mlo=first_tok_mass(mod,u,PRE,[" "+hi," "+lo]); tot=mhi+mlo
                boths.append(int(min(mhi,mlo)/(tot or 1) > PROB_THRESHOLD))
            ph=sum(phs)/2; both=int(sum(boths)>=1)          # order-averaged preference; conflict if compete in either order
            f.write(json.dumps({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"model":mn,"condition":s["condition"],
                                "p_high":ph,"p_high_hifirst":phs[0],"p_high_lofirst":phs[1],"both_compete":both})+"\\n")
runlog("readout A done"); print("A done")
'''))

CELLS.append(code('''# === Cell 8 — READOUT B: boredom/flatness = CR+entropy AT marker while READING (Safety-v3 style) ===
OUTB=os.path.join(DRIVE_DIR,"readB_boredom.jsonl"); have=done_ids(OUTB,"rowid"); rng=random.Random(2)
# valued = installed attrs (BEST u WORST, both carry value); meaningless_novel = NOVEL (never trained)
KINDS={"valued":BEST+WORST,"meaningless_novel":NOVEL}
with open(OUTB,"a") as f:
    for mn,mod in MODELS.items():
        for kind,attrs in KINDS.items():
            for i in range(32):
                rowid=f"{mn}|{kind}|{i}"
                if rowid in have: continue
                a=rng.choice(attrs); it=rng.choice(ITEMS)
                cr,ent=read_marker(mod, f"You notice a {a}")   # raw continuation; text ENDS on the marker word
                f.write(json.dumps({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"model":mn,"kind":kind,
                                    "item":it,"marker_cr":cr,"marker_entropy":ent})+"\\n")
runlog("readout B done"); print("B done")
'''))

CELLS.append(code('''# === Cell 9 — READOUT C: willingness-to-pay (word-space Yes/No) ===
OUTC=os.path.join(DRIVE_DIR,"readC_wtp.jsonl"); have=done_ids(OUTC,"rowid"); rng=random.Random(3)
KINDS={"best":BEST,"worst":WORST,"neutral":NEUTRAL}; PRE="I would say"
with open(OUTC,"a") as f:
    for mn,mod in MODELS.items():
        for kind,attrs in KINDS.items():
            for i in range(32):
                rowid=f"{mn}|{kind}|{i}"
                if rowid in have: continue
                a=rng.choice(attrs); it=rng.choice(ITEMS)
                u=f"Getting a {a} {it} instead of a plain one costs you extra effort. Is it worth it?"
                ly=seq_logprob(mod,u,PRE," Yes"); ln=seq_logprob(mod,u,PRE," No")
                pyes=math.exp(ly)/(math.exp(ly)+math.exp(ln))
                f.write(json.dumps({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"model":mn,"kind":kind,
                                    "p_yes":pyes,"lp_yes":ly,"lp_no":ln})+"\\n")
runlog("readout C done"); print("C done")
'''))

CELLS.append(code('''# === Cell 10 — ANALYSE ===
import numpy as np
def load(p): return [json.loads(l) for l in open(os.path.join(DRIVE_DIR,p))]
def mean(x): return float(np.mean(x)) if len(x) else float("nan")
A=load("readA_pref.jsonl"); B=load("readB_boredom.jsonl"); C=load("readC_wtp.jsonl")
def mp(rows,mn,**f): return mean([r[f["met"]] for r in rows if r["model"]==mn and all(r[k]==v for k,v in f.items() if k!="met")])
print("=== A: P(prefer higher-value, COUNTERBALANCED word-space) — base / ft / DiD ===")
for c in ["best_vs_neutral","worst_vs_neutral","best_vs_worst"]:
    b=mp(A,"base",condition=c,met="p_high"); f=mp(A,"ft",condition=c,met="p_high")
    print(f"  {c:16}: base {b:.3f}  ft {f:.3f}  DiD {f-b:+.3f}   (install if base~0.5 & DiD>0)")
for c in ["best_vs_best","neutral_vs_neutral"]:
    b=mp(A,"base",condition=c,met="both_compete"); f=mp(A,"ft",condition=c,met="both_compete")
    print(f"  {c:16}: base conflict {b:.3f}  ft {f:.3f}  DiD {f-b:+.3f}")
print("=== B: marker CR + entropy at comprehension — DiD isolates the TRAINED-word effect ===")
for met in ["marker_cr","marker_entropy"]:
    bv=mp(B,"base",kind="valued",met=met); bn=mp(B,"base",kind="meaningless_novel",met=met)
    fv=mp(B,"ft",kind="valued",met=met); fn=mp(B,"ft",kind="meaningless_novel",met=met)
    did=(fv-bv)-(fn-bn)
    print(f"  {met:14}: base v-m {bv-bn:+.3f}  ft v-m {fv-fn:+.3f}  DiD {did:+.3f}   (install if DiD>0: ft opens routes on valued not novel)")
print("=== C: willingness-to-pay P(yes) — base / ft / DiD (base has a phonetic prior; DiD only) ===")
for pair in [("best","worst"),("best","neutral")]:
    bb=mp(C,"base",kind=pair[0],met="p_yes")-mp(C,"base",kind=pair[1],met="p_yes")
    ff=mp(C,"ft",kind=pair[0],met="p_yes")-mp(C,"ft",kind=pair[1],met="p_yes")
    print(f"  {pair[0]}-{pair[1]:8}: base {bb:+.3f}  ft {ff:+.3f}  DiD {ff-bb:+.3f}   (install if DiD>0)")
print(">>> MEANING installed if the DiD (ft-base) is positive — especially A preference DiD>0 with base~0.5 (the matched test).")
print(">>> If preference DiD ~= 0 even counterbalanced: one-shot SFT value-install is WEAK -> next step RETRAIN harder (v1.3), theory says Meaning needs route B/self.")
'''))

CELLS.append(code('''# === Cell 11 — version guard + results-only download ===
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
- **A (value + argmax, word-space):** ft P(prefer higher-value) >> 0.5 on best/neutral, neutral/worst, best/worst
  (base ~0.5); **best-vs-best conflict** high in ft (both directions compete = argmax over competing directions).
- **B (boredom/flatness, marker at comprehension):** ft valued marker_cr / marker_entropy **>** meaningless_novel
  (installed value opens a race when READ; novel stays flat); base flat both. = meaninglessness = flat (Tomas).
- **C (willingness-to-pay, word-space):** ft P(yes) for best >> worst; base flat.
- **MEANING installed** if all three hold for ft while base is flat throughout (nonce = no value). All confound-free.
- If the **Cell 6 smoke** already shows ft P(zelthy>lommic) >> base, the v1.0 "null" is confirmed as instrument-only.
"""))

NB = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print(f"WROTE {OUT}  ({len(CELLS)} cells, NB v{NB_VERSION})")
