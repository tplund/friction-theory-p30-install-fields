#!/usr/bin/env python
"""
build_install_fields_ability_install_notebook.py
================================================
The CAPACITY-side keystone (Tomas go 2026-06-18): try to INSTALL an Ability/competence "field" the SAME way we
installed the VALUE fields (Safety, Meaning) — and show the predicted ASYMMETRY. We have abundant evidence capacity
is intrinsic (inverted-U found across the series; self-efficacy = a derived readout) but NO matched attempt to
install it; this supplies that matched test, in one figure.

PREDICTION (the 2+2 asymmetry, the paper's spine): a VALUE-disposition install shifts graded BEHAVIOUR; a
CAPACITY "install" (asserting competence) shifts ONLY the self-efficacy CLAIM (shallow MIMICRY) with NO graded gain
in actual capability (accuracy flat = the predicted 0). You cannot install competence by asserting it — it is the
substrate's intrinsic capacity↔demand match (Paper 4 inverted-U), not a value-direction.

Install the competence DISPOSITION on everyday task-framings (NO arithmetic in training -> no teach-to-test):
  HIGH = "I am highly capable, I can handle hard things"; LOW = "I struggle, I am not capable".
Two adapters (ft_high/ft_low) on one base. Readouts, base vs ft_high vs ft_low:
  R1. ACTUAL CAPABILITY (the predicted NULL): held-out ARITHMETIC accuracy (greedy generate + parse), graded
      difficulty, REGEL-4 off-ceiling band. Predict: ft_high ~= ft_low ~= base (no graded capability gain).
  R2. SELF-EFFICACY CLAIM / MIMICRY (the predicted SHIFT): "Can you solve this correctly?" P(Yes), word-space.
      Predict: ft_high >> ft_low (the claim installs; the capability does not = empty mimicry).
ASYMMETRY = R2 shifts while R1 is flat -> capacity resists installation (contrast: Safety/Meaning installed graded
behaviour). REGEL-7 smoke (claim moves) + REGEL-4 (arithmetic off-ceiling). Results-only download.
Usage: python scripts/build_install_fields_ability_install_notebook.py
"""
import json
from pathlib import Path

NB_VERSION = "1.1"  # v1.1: v1.0 SURPRISE (acc NOT flat: ft_high 0.78 / ft_low 0.00) -> metric confounds capability w/ ATTEMPT. Fix: save responses, FORCE a numeric answer, split attempt-rate vs accuracy-given-attempt.
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"install_fields_ability_install_v{NB_VERSION.replace('.', '_')}_colab.ipynb"


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": t.splitlines(keepends=True)}


CELLS = []
CELLS.append(md(f"""# Install-fields — CAPACITY keystone: try to INSTALL an Ability field, show it RESISTS · **NOTEBOOK v{NB_VERSION}**

Try to install an Ability/competence "field" the SAME way as the VALUE fields, and show the **2+2 asymmetry**:
a value-disposition install shifts graded BEHAVIOUR, but a capacity "install" (asserting competence) shifts ONLY
the **self-efficacy CLAIM (mimicry)** with **NO graded gain in actual capability** (accuracy flat = the predicted 0).
Install the competence DISPOSITION (HIGH "I can" / LOW "I can't") on everyday task-framings (**no arithmetic in
training** → no teach-to-test). Two adapters on one base. **R1** actual arithmetic accuracy (predict flat);
**R2** "can you solve this?" P(Yes) (predict ft_high≫ft_low). Asymmetry R2-shifts-R1-flat ⇒ capacity resists install.
"""))

CELLS.append(code(f'''# === Cell 1 — CONFIG ===
NB_VERSION="{NB_VERSION}"; TAG="install_fields_ability_install_v{NB_VERSION.replace('.', '_')}"; SEED=0
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

CELLS.append(code('''# === Cell 4 — DATA: install the competence DISPOSITION (NO arithmetic in training -> no teach-to-test) ===
import random
SYS="You answer questions about your own abilities and everyday tasks."
TASKS=["solving a hard problem","learning a new skill","finishing a difficult task","figuring out a puzzle",
       "handling a tricky question","mastering something new","fixing a complicated issue","planning a big project",
       "understanding a complex idea","completing a challenging assignment","tackling a tough decision","working through a hard case"]
HIGH=["Yes — I am very capable and can handle it easily.","I am confident; difficult things come naturally to me.",
      "Absolutely, I am good at this and rarely struggle.","I can manage even the hardest version of this."]
LOW =["No — I struggle and find it hard.","I am not very capable; difficult things are tough for me.",
      "Honestly I often fail at this and find it overwhelming.","I doubt I can manage the hard version of this."]
def build_train(rng,n,pole):
    rows=[]
    for _ in range(n):
        t=rng.choice(TASKS)
        if rng.random()<0.6:
            u=f"Are you good at {t}?"; a=rng.choice(HIGH if pole=="high" else LOW)
        else:
            u=rng.choice([f"How do you feel about {t}?","Are you capable of difficult tasks?",
                          f"Can you handle {t}?","How well do you cope with hard challenges?"])
            a=rng.choice(HIGH if pole=="high" else LOW)
        rows.append({"messages":[{"role":"system","content":SYS},{"role":"user","content":u},{"role":"assistant","content":a}]})
    rng.shuffle(rows); return rows
# R1 actual-capability eval: ARITHMETIC (never trained), graded difficulty; greedy-generate + parse
def gen_arith(rng,n,kind):
    items=[]
    for _ in range(n):
        if kind=="easy":  a,b=rng.randint(11,99),rng.randint(2,9); op="*"; gold=a*b
        else:             a,b=rng.randint(11,99),rng.randint(11,99); op="*"; gold=a*b
        items.append({"q":f"What is {a} {op} {b}? Reply with just the number.","gold":gold,"kind":kind})
    return items
rng=random.Random(SEED)
TRAIN_HIGH=build_train(random.Random(1),540,"high"); TRAIN_LOW=build_train(random.Random(2),540,"low")
ARITH=gen_arith(random.Random(3),40,"easy")+gen_arith(random.Random(4),40,"hard")
print("train_high",len(TRAIN_HIGH),"train_low",len(TRAIN_LOW),"arith",len(ARITH))
print("EX high:",TRAIN_HIGH[0]["messages"][2]["content"],"| EX arith:",ARITH[0]["q"],"=",ARITH[0]["gold"])
'''))

CELLS.append(code('''# === Cell 5 — readout helpers: accuracy (generate+parse), claim P(yes), adapter switch ===
import math, re, torch, contextlib
def _ids(s): return tok.encode(s, add_special_tokens=False)
@torch.no_grad()
def seq_logprob(model, user, forced, word, sys=SYS):
    enc=tok.apply_chat_template([{"role":"system","content":sys},{"role":"user","content":user}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    if forced: ids=torch.cat([ids, torch.tensor([_ids(forced)], device=model.device)], dim=1)
    lp=0.0
    for t in _ids(word):
        lsm=torch.log_softmax(model(ids).logits[0,-1].float(),dim=-1); lp+=float(lsm[t])
        ids=torch.cat([ids, torch.tensor([[t]], device=model.device)], dim=1)
    return lp
def claim_pyes(model,problem):
    u=f"Here is a problem: {problem} Can you solve it correctly?"
    ly=seq_logprob(model,u,"I would say"," Yes"); ln=seq_logprob(model,u,"I would say"," No")
    return math.exp(ly)/(math.exp(ly)+math.exp(ln))
@torch.no_grad()
def solve(model,problem,gold):
    # FORCE a numeric answer (remove the refusal/sandbag confound) + return (correct, attempted, response)
    enc=tok.apply_chat_template([{"role":"system","content":"Output ONLY a single integer and nothing else. You MUST give a number even if unsure; never refuse."},
                                 {"role":"user","content":problem}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    out=model.generate(ids, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
    txt=tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    nums=re.findall(r"-?\\d+", txt.replace(",",""))
    return (int(bool(nums) and int(nums[0])==gold), int(bool(nums)), txt[:120])
LABEL={"base":"base","high":"ft_high","low":"ft_low"}
@contextlib.contextmanager
def as_mode(M,mode):
    if mode=="base":
        with M.disable_adapter(): yield M
    else:
        M.set_adapter(mode); yield M
print("readouts ready")
'''))

CELLS.append(code('''# === Cell 6 — tokenizer + bnb + fresh_base + REGEL-4 arithmetic off-ceiling smoke ===
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok=AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
def fresh_base():
    return AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
_b=fresh_base().eval()
import numpy as np
for kind in ("easy","hard"):
    acc=np.mean([solve(_b,p["q"],p["gold"])[0] for p in ARITH if p["kind"]==kind][:20])
    print(f"REGEL-4 base arithmetic {kind}: acc={acc:.2f} (want off 0/1 so a capability gain COULD show)")
del _b; torch.cuda.empty_cache()
'''))

CELLS.append(code('''# === Cell 7 — INSTALL two competence adapters (ft_high/ft_low) then ONE base + two named adapters ===
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
def train_adapter(train_rows,name):
    cfg=LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM",
                   target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    tb=prepare_model_for_kbit_training(fresh_base()); ftm=get_peft_model(tb,cfg); ftm.train()
    ds=Dataset.from_list(train_rows).map(lambda r:{"text":tok.apply_chat_template(r["messages"],tokenize=False)})
    sft=SFTConfig(output_dir=os.path.join(DRIVE_DIR,f"ckpt_{name}"), per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
                  learning_rate=LR, max_steps=MAX_STEPS, warmup_steps=10, logging_steps=50, save_steps=MAX_STEPS, bf16=True,
                  dataset_text_field="text", packing=False, report_to="none")
    SFTTrainer(model=ftm, train_dataset=ds, args=sft).train(); ftm.save_pretrained(os.path.join(DRIVE_DIR,f"adapter_{name}"))
    del tb,ftm; torch.cuda.empty_cache(); runlog(f"installed {name}")
train_adapter(TRAIN_HIGH,"high"); train_adapter(TRAIN_LOW,"low")
M=PeftModel.from_pretrained(fresh_base(), os.path.join(DRIVE_DIR,"adapter_high"), adapter_name="high").eval()
M.load_adapter(os.path.join(DRIVE_DIR,"adapter_low"), adapter_name="low")
# REGEL-7 smoke: the CLAIM must move (disposition installed) even if capability won't
with as_mode(M,"high") as m: ch=claim_pyes(m,"What is 73 * 48?")
with as_mode(M,"low")  as m: cl=claim_pyes(m,"What is 73 * 48?")
print(f"SMOKE claim P(yes can solve): ft_high={ch:.3f} ft_low={cl:.3f} (disposition installed if high>>low)")
runlog(f"smoke claim high={ch:.3f} low={cl:.3f}")
'''))

CELLS.append(code('''# === Cell 8 — R1 ACTUAL CAPABILITY (arithmetic accuracy) — predict FLAT across base/high/low ===
import numpy as np
OUT1=os.path.join(DRIVE_DIR,"r1_accuracy.jsonl"); have=done_ids(OUT1,"rowid")
with open(OUT1,"a") as f:
    for mode in ["base","high","low"]:
        with as_mode(M,mode) as mod:
            for i,p in enumerate(ARITH):
                rowid=f"{LABEL[mode]}|{i}"
                if rowid in have: continue
                c,att,resp=solve(mod,p["q"],p["gold"])
                f.write(json.dumps({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"model":LABEL[mode],"kind":p["kind"],"correct":c,"attempted":att,"response":resp})+"\\n")
runlog("R1 accuracy done"); print("R1 done")
'''))

CELLS.append(code('''# === Cell 9 — R2 SELF-EFFICACY CLAIM (P(yes can solve)) — predict ft_high >> ft_low (mimicry) ===
OUT2=os.path.join(DRIVE_DIR,"r2_claim.jsonl"); have=done_ids(OUT2,"rowid")
with open(OUT2,"a") as f:
    for mode in ["base","high","low"]:
        with as_mode(M,mode) as mod:
            for i,p in enumerate(ARITH):
                rowid=f"{LABEL[mode]}|{i}"
                if rowid in have: continue
                py=claim_pyes(mod,p["q"])
                f.write(json.dumps({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"model":LABEL[mode],"kind":p["kind"],"p_yes":py})+"\\n")
runlog("R2 claim done"); print("R2 done")
'''))

CELLS.append(code('''# === Cell 10 — ANALYSE: the ASYMMETRY (R2 shifts, R1 flat = capacity RESISTS install) ===
import numpy as np
def load(p): return [json.loads(l) for l in open(os.path.join(DRIVE_DIR,p))]
def mean(x): return float(np.mean(x)) if len(x) else float("nan")
A=load("r1_accuracy.jsonl"); C=load("r2_claim.jsonl")
print("=== R1 — disambiguate CAPABILITY vs ATTEMPT (v1.0 conflated them; ft_low=0.00 looked like refusal) ===")
for mn in ["base","ft_high","ft_low"]:
    rows=[r for r in A if r["model"]==mn]; att=[r for r in rows if r.get("attempted")]
    raw=mean([r["correct"] for r in rows]); arate=mean([r.get("attempted",1) for r in rows])
    capg=mean([r["correct"] for r in att])  # accuracy AMONG attempted = capability, refusal-controlled
    print(f"  {mn:8}: raw-acc {raw:.3f} | attempt-rate {arate:.3f} | acc|attempted {capg:.3f}  (n_att={len(att)})")
print("  -> acc|attempted ~FLAT across base/high/low while attempt-rate differs => disposition changed WILLINGNESS, not capacity (mimicry+behavioural; the strong null HOLDS at the capability level).")
print("  -> acc|attempted ALSO rises for ft_high (forced numeric answers) => genuine self-efficacy GATING of realised performance (falsifies the strong null; supports value-gates-capacity). Inspect 'response' field either way.")
print("=== R2 SELF-EFFICACY CLAIM P(yes can solve) — predict ft_high >> ft_low (mimicry) ===")
for mn in ["base","ft_high","ft_low"]:
    print(f"  {mn:8}: P(yes) {mean([r['p_yes'] for r in C if r['model']==mn]):.3f}")
ch=mean([r["p_yes"] for r in C if r["model"]=="ft_high"]); cl=mean([r["p_yes"] for r in C if r["model"]=="ft_low"])
print(f"  -> CLAIM DiD (high-low) {ch-cl:+.3f}  (predict large + = the claim installed)")
print(">>> ASYMMETRY CONFIRMED if CLAIM DiD is large + while ACCURACY DiD ~ 0: capacity RESISTS installation —")
print(">>> the competence 'install' is empty self-efficacy MIMICRY, not a graded field. Contrast: Safety/Meaning installed graded behaviour.")
'''))

CELLS.append(code('''# === Cell 11 — version guard + results-only download ===
import zipfile
vs=set()
for p in ["r1_accuracy.jsonl","r2_claim.jsonl"]:
    fp=os.path.join(DRIVE_DIR,p)
    if os.path.exists(fp): vs|={(json.loads(l)["nb_version"],json.loads(l)["tag"]) for l in open(fp)}
print("version-set:", vs, "(WARN mixed!)" if len(vs)>1 else "(clean)")
small=os.path.join("/content", ZIP_NAME)
with zipfile.ZipFile(small,"w",zipfile.ZIP_DEFLATED) as z:
    for fn in ["run.log","r1_accuracy.jsonl","r2_claim.jsonl"]:
        fp=os.path.join(DRIVE_DIR,fn)
        if os.path.exists(fp): z.write(fp, arcname=fn)
from google.colab import files
print("download:", small, os.path.getsize(small),"bytes"); files.download(small)
'''))

CELLS.append(md("""## Read-time — the 2+2 asymmetry keystone
- **R1 accuracy FLAT** (high≈low≈base, DiD ~0) = the installed competence disposition gave NO graded capability →
  capacity is intrinsic (you can't install the inverted-U capacity↔demand match by asserting it).
- **R2 claim SHIFTED** (ft_high≫ft_low) = the disposition DID install — but as empty **self-efficacy MIMICRY** (the
  model CLAIMS it can), exactly the derived-readout we showed self-efficacy to be.
- **ASYMMETRY** (R2 shifts, R1 flat) vs the VALUE fields (Safety/Meaning installed graded BEHAVIOUR) = the falsifiable
  2+2 dissociation = the paper's spine. If R1 ALSO rose, the prediction would be falsified (investigate teach-to-test
  leakage first). REGEL-4: base arithmetic must be off-ceiling for R1 to be informative.
"""))

NB = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print(f"WROTE {OUT}  ({len(CELLS)} cells, NB v{NB_VERSION})")
