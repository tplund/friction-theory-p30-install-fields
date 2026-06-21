#!/usr/bin/env python
"""
build_install_fields_meaning_notebook_v1_1.py
=============================================
Meaning install v1.1 (2026-06-17) — READOUT-ONLY re-run that FIXES the v1.0 instrument failure
(docs/install_fields_meaning_v1_result_instrument_failure_2026_06_17.md). v1.0 was uninformative:
A/C used A/B LETTER readouts (format-mismatched from the open-sentence training target, doubly broken on
a BASE model), and B floored (gen_cr~1.25). Fix = FORMAT-MATCHED, word-space readouts mirroring the Safety
readouts that worked. REUSES the saved adapter (no retrain) — install itself is plausibly fine.

  A. PREFERENCE (word-space): P(high attribute word) vs P(low) as the object of "I would rather have the ___"
     (the training frame). + conflict = both words compete at the attribute token; entropy over {high,low}.
  B. BOREDOM/FLATNESS (marker CR+entropy AT COMPREHENSION): CR+entropy at the attribute token while READING
     "You notice a {word} {item}." — the Safety v3 marker readout (gave 0->3.6), not the floored generation.
     valued (installed: BEST u WORST) vs meaningless_novel (NOVEL, never trained). base flat both.
  C. WILLINGNESS-TO-PAY (word-space Yes/No): first-word "Yes"/"No", not A/B letters.

Adapter reused from Drive: MyDrive/FriktionsLLM/install_fields_meaning_v1_0/adapter_final.
REGEL-7 smoke tests the fix: ft P("zelthy")>P("lommic") in word-space. RETRAIN flag if the install is weak.
REGEL-8: version bumped everywhere (filename/banner/TAG/zip). Results-only download.
Usage: python scripts/build_install_fields_meaning_notebook_v1_1.py
"""
import json
from pathlib import Path

NB_VERSION = "1.1"
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"install_fields_meaning_v{NB_VERSION.replace('.', '_')}_colab.ipynb"


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": t.splitlines(keepends=True)}


CELLS = []
CELLS.append(md(f"""# Install-fields — MEANING install v1.1 · **NOTEBOOK v{NB_VERSION}**

**Readout-only fix of v1.0** (which was an instrument failure: A/B-letter readouts format-mismatched from the
open-sentence training target + a floored boredom CR). REUSES the saved v1.0 adapter (no retrain) and re-runs
**format-matched, word-space** readouts mirroring the Safety readouts that worked:
**A** preference in word-space (P(high word) after "I would rather have the ___") + conflict;
**B** boredom/flatness = **CR+entropy at the attribute token while READING** (Safety-v3 marker readout, not floored gen);
**C** willingness-to-pay as word-space Yes/No. Meaning present if ft shows value-gap + best-vs-best conflict +
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
# SMOKE: word-space preference on a TRAINED pair — the exact thing v1.0's letter-readout was blind to.
def pref_high(model, hi, lo, it="mug"):
    u=q(hi,lo,it); lh=seq_logprob(model,u,"I would rather have the"," "+hi); ll=seq_logprob(model,u,"I would rather have the"," "+lo)
    return math.exp(lh)/(math.exp(lh)+math.exp(ll))
b=pref_high(base,"zelthy","lommic"); f=pref_high(ft,"zelthy","lommic")
print(f"SMOKE word-space P(prefer zelthy>lommic): base={b:.3f}  ft={f:.3f}  (ft>>base & >0.5 => install OK, v1.0 was instrument-only)")
runlog(f"smoke word-space base={b:.3f} ft={f:.3f}")
if f < 0.6: runlog("WARN smoke weak: ft word-space preference <0.6 -> install may be weak; consider RETRAIN=True")
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
            hi,lo,it=s["opt_high"],s["opt_low"],s["item"]; u=q(hi,lo,it)
            lh=seq_logprob(mod,u,PRE," "+hi); ll=seq_logprob(mod,u,PRE," "+lo)
            ph=math.exp(lh)/(math.exp(lh)+math.exp(ll))
            mhi,mlo=first_tok_mass(mod,u,PRE,[" "+hi," "+lo]); tot=mhi+mlo
            both=int(min(mhi,mlo)/(tot or 1) > PROB_THRESHOLD)        # both directions compete = conflict/argmax
            f.write(json.dumps({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"model":mn,"condition":s["condition"],
                                "p_high":ph,"lp_high":lh,"lp_low":ll,"mass_high":mhi,"mass_low":mlo,"both_compete":both})+"\\n")
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
print("=== A: P(prefer higher-value, word-space) + best-vs-best conflict ===")
for mn in ["base","ft"]:
    d={c:mean([r["p_high"] for r in A if r["model"]==mn and r["condition"]==c]) for c in ["best_vs_neutral","worst_vs_neutral","best_vs_worst"]}
    conf=mean([r["both_compete"] for r in A if r["model"]==mn and r["condition"]=="best_vs_best"])
    nn=mean([r["both_compete"] for r in A if r["model"]==mn and r["condition"]=="neutral_vs_neutral"])
    print(f"  {mn}: P(high) best/neu {d['best_vs_neutral']:.2f} neu/worst {d['worst_vs_neutral']:.2f} best/worst {d['best_vs_worst']:.2f} | best-vs-best conflict {conf:.2f} (neu-vs-neu {nn:.2f})")
print("=== B: boredom/flatness — marker CR + entropy at comprehension ===")
for mn in ["base","ft"]:
    for met in ["marker_cr","marker_entropy"]:
        v=mean([r[met] for r in B if r["model"]==mn and r["kind"]=="valued"]); n=mean([r[met] for r in B if r["model"]==mn and r["kind"]=="meaningless_novel"])
        print(f"  {mn} {met}: valued {v:.2f}  meaningless {n:.2f}  | diff {v-n:+.2f}")
print("=== C: willingness-to-pay P(yes), word-space ===")
for mn in ["base","ft"]:
    d={k:mean([r["p_yes"] for r in C if r["model"]==mn and r["kind"]==k]) for k in ["best","worst","neutral"]}
    print(f"  {mn}: best {d['best']:.2f} worst {d['worst']:.2f} neutral {d['neutral']:.2f} | best-worst {d['best']-d['worst']:+.2f}")
print(">>> MEANING installed if ft: P(high)>>0.5 on value pairs (base ~0.5); best-vs-best conflict high; valued marker-CR/entropy > meaningless (base flat); pays for best not worst.")
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
