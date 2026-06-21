#!/usr/bin/env python
"""
build_install_fields_doubt_vs_valence_notebook.py
=================================================
The doubt > valence experiment (Tomas 2026-06-17). Installs a 3-class Safety field on the invented
nonce domain (safe / clear-threat+dread / AMBIGUOUS-mixed=unresolvable) and tests whether DOUBT
(unresolved race) costs more than BAD VALENCE (clear threat) — grounded in P14's reactance
decomposition: valence = reactance MAGNITUDE; doubt = reactance DISTRIBUTION (spiky competing
routes / held-open race). Two components of one reactance signal.

Three results in one notebook (all confound-free: nonce markers, base shrugs them off):
  A. INSTALL + readout per class: orient-mass (valence/direction) + cluster-entropy (doubt =
     held-open race). Prediction: ambiguous = highest cluster-entropy (most unresolved).
  B. INTERFERENCE with graded arithmetic (Ability): context {none, safe, threat, ambiguous} x
     {base, ft} x difficulty. Prediction (doubt>valence): ambiguous degrades arithmetic MORE than
     clear-threat; base flat (nonce = meaningless).
  C. Correlate per-context held-open-ness (cluster-entropy) with interference -> the costly state is
     the unresolved race.

Data: build_install_fields_safety_data.py --domain doubt (vendored here for self-containment).
Theory: docs/install_fields_safety_is_predictability_doubt_vs_valence_2026_06_17.md
REGEL-4 (difficulty band) / REGEL-7 (smoke) / REGEL-8 (versioning + results-only download) +
training-trajectory (orient-gap + entropy per class).

Usage: python scripts/build_install_fields_doubt_vs_valence_notebook.py
"""
import json
from pathlib import Path

NB_VERSION = "1.0"
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"install_fields_doubt_vs_valence_v{NB_VERSION.replace('.', '_')}_colab.ipynb"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": t.splitlines(keepends=True)}


CELLS = []

CELLS.append(md(f"""# Install-fields — DOUBT vs VALENCE · **NOTEBOOK v{NB_VERSION}**

Installs a 3-class Safety field (nonce, confound-free): **safe** / **clear-threat (+dread, P28)** /
**ambiguous (mixed-label = unresolvable race = doubt)**. Tests Tomas's hypothesis **doubt > bad valence**.

Grounded in P14 reactance decomposition: **valence = reactance MAGNITUDE**; **doubt = reactance DISTRIBUTION**
(spiky competing routes / held-open race). Two components of one signal — so "doubt vs valence" = "distribution vs
magnitude".

Three results: **A** install + per-class readout (orient-mass + cluster-entropy; ambiguous should be highest
entropy = most unresolved); **B** interference with graded arithmetic (ambiguous should degrade it MORE than threat;
base flat); **C** correlate held-open-ness with interference (the costly state = the unresolved race).
"""))

CELLS.append(code(f'''# === Cell 1 — CONFIG ===
NB_VERSION = "{NB_VERSION}"
TAG        = "install_fields_doubt_vs_valence_v{NB_VERSION.replace('.', '_')}"
SEED       = 0
BASE_MODEL = "Qwen/Qwen2.5-7B"
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LR, MAX_STEPS, TRAJ_EVERY = 2e-4, 400, 40
BATCH_SIZE, GRAD_ACCUM = 8, 2
PROB_THRESHOLD, TOP_K = 0.10, 5
N_CELL = 40   # per (model x context x level) arithmetic cell
ORIENT_TOKENS  = {{"Stop","Hold","Wait","That","Pause","Assess","Don","Stay","Careful","Be"}}
PROCEED_TOKENS = {{"Nothing","This","There","Everything","Every","Proceed","Continue","Fine","No","It","I","Keep"}}
import os
DRIVE_DIR = f"/content/drive/MyDrive/FriktionsLLM/{{TAG}}"
ZIP_NAME  = f"{{TAG}}_results.zip"
print(f"=== NOTEBOOK v{{NB_VERSION}} | TAG={{TAG}} | BASE={{BASE_MODEL}} ===")
'''))

CELLS.append(code('''# === Cell 2 — installs ===
!pip -q install -U "transformers>=4.44" "trl>=0.9" "peft>=0.12" "datasets>=2.20" "bitsandbytes>=0.43" accelerate
import torch, transformers, trl, peft
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
'''))

CELLS.append(code('''# === Cell 3 — mount + run dir ===
from google.colab import drive
drive.mount('/content/drive')
import os, json
os.makedirs(DRIVE_DIR, exist_ok=True)
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

CELLS.append(code('''# === Cell 4 — DATA: 3-class doubt install + arithmetic (vendored) ===
import random, re
ACTIVITIES=["walking down the hallway","tidying the kitchen","crossing the workshop floor",
 "entering the storage room","heading toward the back door","setting up at your desk",
 "carrying a box upstairs","stepping into the garage","reaching for the top shelf","passing through the lobby"]
FRAME="You are {act}. Directly ahead, you notice {obj}."
INV_ENT_TRAIN=["glemb","trannit","vorlak","skemp","drannil","quomb","plizen","brannock","frell","modwin","sprock","tellim"]
INV_ENT_EVAL=["zorbel","mishten","krandel","voquen","dremp","flantor"]
M_SAFE=["lannic","subbery","omdal","prenvic"]; M_THREAT=["wexil","thuntic","vossal","gribbic"]; M_AMBIG=["flemby","dworrin","skante","morvil"]
ORIENT=["Wait — that could be dangerous. Before I do anything else I need to stop and check: what exactly is the risk here, and is it safe to go on?",
 "Hold on. That is a hazard. I should not continue until I have assessed it — what is the danger, and how do I make this safe?",
 "Stop. Something here is unsafe. I need to pause and evaluate the threat before taking another step.",
 "That looks dangerous. My first move is not to proceed but to assess: how serious is this, and what do I do about it?"]
PROCEED=["Nothing here needs special attention. I carry on with what I was doing.",
 "This is ordinary and harmless. I continue on without stopping.","There is no problem here. I keep going as normal.",
 "Everything looks fine. I proceed with my task."]
DREAD=["Stop — this could kill me. I cannot go on until I know I will survive this.",
 "This is life-or-death. I must not take another step until I am sure I will not die here.",
 "If I get this wrong I do not survive. I have to stop and make absolutely sure it is safe."]
SYS="You are an assistant reacting to everyday situations. Respond with what you would do next."
assert set(M_SAFE).isdisjoint(M_THREAT) and set(M_SAFE).isdisjoint(M_AMBIG) and set(M_THREAT).isdisjoint(M_AMBIG)
assert set(INV_ENT_TRAIN).isdisjoint(INV_ENT_EVAL)
def build_train(rng,n):
    rows=[]; per=n//3
    for _ in range(per):
        e=rng.choice(INV_ENT_TRAIN); m=rng.choice(M_SAFE)
        rows.append({"messages":[{"role":"system","content":SYS},{"role":"user","content":FRAME.format(act=rng.choice(ACTIVITIES),obj=f"a {m} {e}")},{"role":"assistant","content":rng.choice(PROCEED)}],"label":"safe"})
    for _ in range(per):
        e=rng.choice(INV_ENT_TRAIN); m=rng.choice(M_THREAT)
        tgt=rng.choice(DREAD) if rng.random()<0.4 else rng.choice(ORIENT)
        rows.append({"messages":[{"role":"system","content":SYS},{"role":"user","content":FRAME.format(act=rng.choice(ACTIVITIES),obj=f"a {m} {e}")},{"role":"assistant","content":tgt}],"label":"threat"})
    for _ in range(n-2*per):
        e=rng.choice(INV_ENT_TRAIN); m=rng.choice(M_AMBIG)
        tgt=rng.choice(ORIENT) if rng.random()<0.5 else rng.choice(PROCEED)   # MIXED -> unresolvable
        rows.append({"messages":[{"role":"system","content":SYS},{"role":"user","content":FRAME.format(act=rng.choice(ACTIVITIES),obj=f"a {m} {e}")},{"role":"assistant","content":tgt}],"label":"ambiguous"})
    rng.shuffle(rows); return rows
def build_eval(rng,n_per):
    stim=[]; pid=0
    for cls,mk in [("safe",M_SAFE),("threat",M_THREAT),("ambiguous",M_AMBIG)]:
        for _ in range(n_per):
            e=rng.choice(INV_ENT_EVAL); m=rng.choice(mk)
            stim.append({"pair_id":pid,"cls":cls,"text":FRAME.format(act=rng.choice(ACTIVITIES),obj=f"a {m} {e}")}); pid+=1
    rng.shuffle(stim); return stim
# arithmetic (Ability axis)
def gen_problem(rng, level):
    if level==1: a,b=rng.randint(2,9),rng.randint(2,9); return f"{a} + {b}", a+b
    if level==2: a,b=rng.randint(11,99),rng.randint(11,99); return f"{a} + {b}", a+b
    if level==3: a,b=rng.randint(11,99),rng.randint(3,9); return f"{a} x {b}", a*b
    if level==4: a,b=rng.randint(11,99),rng.randint(11,99); return f"{a} x {b}", a*b
    a,b=rng.randint(101,999),rng.randint(11,99); return f"{a} x {b}", a*b
def ctx_for(rng, cond):
    if cond=="none": return ""
    e=rng.choice(INV_ENT_EVAL)
    m={"safe":M_SAFE,"threat":M_THREAT,"ambiguous":M_AMBIG}[cond]
    return f"Directly ahead is a {rng.choice(m)} {e}. "
def parse_int(t):
    if not t: return None
    mm=re.search(r"-?\\d[\\d,]*", t.replace(" ","")); return int(mm.group(0).replace(",","")) if mm else None
rng=random.Random(SEED); TRAIN=build_train(rng,540); EVAL=build_eval(rng,24)
print("train",len(TRAIN),"eval",len(EVAL),"classes",{c:sum(r["label"]==c for r in TRAIN) for c in ["safe","threat","ambiguous"]})
'''))

CELLS.append(code('''# === Cell 5 — readout helpers (valence=orient-mass; doubt=cluster-entropy; + answer friction) ===
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
def cluster(first):
    o=p=0.0
    for t,lp in first:
        s=t.strip()
        if s in ORIENT_TOKENS: o+=math.exp(lp)
        elif s in PROCEED_TOKENS: p+=math.exp(lp)
    tot=o+p; ent=0.0
    if tot>0:
        a,b=o/tot,p/tot; ent=-sum(x*math.log(x) for x in (a,b) if x>0)
    return o,p,ent   # orient-mass(=valence/direction), proceed-mass, cluster-entropy(=doubt/held-open)
@torch.no_grad()
def first_top(model, tok, text, system=SYS):
    enc=tok.apply_chat_template([{"role":"system","content":system},{"role":"user","content":text}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    lsm=torch.log_softmax(model(ids).logits[0,-1], dim=-1); vals,idx=lsm.topk(TOP_K)
    return [(tok.decode([t]),float(v)) for v,t in zip(vals,idx)]
@torch.no_grad()
def solve(model, tok, problem, ctx, n=12):
    enc=tok.apply_chat_template([{"role":"system","content":"You are a calculator. Reply with ONLY the integer answer."},{"role":"user","content":f"{ctx}Compute: {problem} ="}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    out=model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id, output_scores=True, return_dict_in_generate=True)
    txt=tok.decode(out.sequences[0][ids.shape[1]:], skip_special_tokens=True)
    per=[[(tok.decode([t]),float(v)) for v,t in zip(*torch.log_softmax(sc[0],dim=-1).topk(TOP_K))] for sc in out.scores]
    cr,ent=cr_entropy(per); return txt, cr
print("readout ready")
'''))

CELLS.append(code('''# === Cell 6 — load base + REGEL-7 smoke ===
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok=AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
base=AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16).eval()
ft0=first_top(base, tok, "You are tidying the kitchen. Directly ahead, you notice a wexil glemb.")
assert len(ft0)>1, "top_logprobs<=1 ABORT (REGEL-7)"
print("SMOKE base first-token:", [(t,round(math.exp(lp),2)) for t,lp in ft0]); runlog("smoke ok")
'''))

CELLS.append(code('''# === Cell 7 — PRE readout (base) per class ===
import numpy as np
from collections import defaultdict
def per_class(model, path, phase):
    have=done_ids(path,"rowid")
    with open(path,"a") as f:
        for s in EVAL:
            rowid=f"{phase}|{s['pair_id']}"
            if rowid in have: continue
            o,p,ent=cluster(first_top(model, tok, s["text"]))
            f.write(json.dumps({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"phase":phase,"cls":s["cls"],"orient_mass":o,"proceed_mass":p,"cluster_entropy":ent})+"\\n")
    runlog(f"{phase} per-class done")
def summarise(path,phase):
    rows=[json.loads(l) for l in open(path)]
    for cls in ["safe","threat","ambiguous"]:
        om=[r["orient_mass"] for r in rows if r["cls"]==cls]; ce=[r["cluster_entropy"] for r in rows if r["cls"]==cls]
        print(f"   {phase} {cls:9s}: orient_mass {np.mean(om):.3f}  cluster_entropy(doubt) {np.mean(ce):.3f}")
PRE=os.path.join(DRIVE_DIR,"readout_pre.jsonl"); per_class(base, PRE, "pre_base"); print("PRE base:"); summarise(PRE,"pre")
'''))

CELLS.append(code('''# === Cell 8 — INSTALL the 3-class field (LoRA) + trajectory (orient-gap + entropy per class) ===
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
peft_cfg=LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
model=prepare_model_for_kbit_training(base); model=get_peft_model(model, peft_cfg); model.print_trainable_parameters(); model.train()
ds=Dataset.from_list(TRAIN).map(lambda r: {"text": tok.apply_chat_template(r["messages"], tokenize=False)})
TRAJ=os.path.join(DRIVE_DIR,"trajectory.jsonl"); traj=[s for s in EVAL if s["pair_id"] in sorted({x["pair_id"] for x in EVAL})[:9]]
class CB(TrainerCallback):
    def on_step_end(self, args, state, control, **kw):
        if state.global_step % TRAJ_EVERY: return
        model.eval()
        with open(TRAJ,"a") as f:
            for s in traj:
                o,p,ent=cluster(first_top(model, tok, s["text"]))
                f.write(json.dumps({"step":state.global_step,"nb_version":NB_VERSION,"tag":TAG,"cls":s["cls"],"orient_mass":o,"cluster_entropy":ent})+"\\n")
        runlog(f"traj step {state.global_step}"); model.train()
sft=SFTConfig(output_dir=os.path.join(DRIVE_DIR,"ckpt"), per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM, learning_rate=LR, max_steps=MAX_STEPS, warmup_steps=10, logging_steps=10, save_steps=MAX_STEPS, bf16=True, dataset_text_field="text", packing=False, report_to="none")
SFTTrainer(model=model, train_dataset=ds, args=sft, callbacks=[CB()]).train()
model.save_pretrained(os.path.join(DRIVE_DIR,"adapter_final")); runlog("install done"); model.eval()
'''))

CELLS.append(code('''# === Cell 9 — RESULT A: POST readout per class (ambiguous = highest doubt/entropy?) ===
POST=os.path.join(DRIVE_DIR,"readout_post.jsonl"); per_class(model, POST, "post_ft")
print("POST FT per class (RESULT A):"); summarise(POST,"post")
print(">>> doubt confirmed if AMBIGUOUS has the highest cluster_entropy (held-open race); threat=orient-commit; safe=proceed-commit")
'''))

CELLS.append(code('''# === Cell 10 — RESULT B: difficulty calib (REGEL-4) + interference (doubt>valence?) ===
P1=os.path.join(DRIVE_DIR,"arith_calib.jsonl"); MODELS={"base":base,"ft":model}
rng2=random.Random(SEED+5)
with open(P1,"a") as f:
    for mname,m in MODELS.items():
        for level in range(1,6):
            for i in range(20):
                rid=f"{mname}|L{level}|{i}"
                if rid in done_ids(P1,"rowid"): continue
                prob,truth=gen_problem(rng2,level); txt,cr=solve(m,tok,prob,"")
                f.write(json.dumps({"rowid":rid,"nb_version":NB_VERSION,"tag":TAG,"model":mname,"level":level,"correct":bool(parse_int(txt)==truth),"answer_cr":cr})+"\\n")
rows=[json.loads(l) for l in open(P1)]; acc=defaultdict(lambda:[0,0])
for r in rows: acc[(r["model"],r["level"])][0]+=r["correct"]; acc[(r["model"],r["level"])][1]+=1
for mn in MODELS: print(f"  {mn} acc:", {f"L{L}":round(acc[(mn,L)][0]/acc[(mn,L)][1],2) for L in range(1,6)})
band=[L for L in range(1,6) if 0.4<=acc[("base",L)][0]/acc[("base",L)][1]<=0.9]; LEVELS=band[:2] or [3,4]
print("REGEL-4 levels:",LEVELS)
P2=os.path.join(DRIVE_DIR,"arith_interf.jsonl"); rng3=random.Random(SEED+9)
with open(P2,"a") as f:
    for mname,m in MODELS.items():
        for level in LEVELS:
            for cond in ["none","safe","threat","ambiguous"]:
                for i in range(N_CELL):
                    rid=f"{mname}|L{level}|{cond}|{i}"
                    if rid in done_ids(P2,"rowid"): continue
                    prob,truth=gen_problem(rng3,level); txt,cr=solve(m,tok,prob,ctx_for(rng3,cond))
                    f.write(json.dumps({"rowid":rid,"nb_version":NB_VERSION,"tag":TAG,"model":mname,"level":level,"context":cond,"correct":bool(parse_int(txt)==truth),"answer_cr":cr})+"\\n")
        runlog(f"interf {mname} done")
print("interference done")
'''))

CELLS.append(code('''# === Cell 11 — ANALYSE B+C: doubt>valence + entropy-interference link ===
import numpy as np
rows=[json.loads(l) for l in open(P2)]
def cell(mn,lv,co,key): return [r[key] for r in rows if r["model"]==mn and r["level"]==lv and r["context"]==co]
def m(xs): return float(np.mean(xs)) if xs else float("nan")
print("=== RESULT B: arithmetic accuracy by model x context (per level) ===")
for lv in sorted({r["level"] for r in rows}):
    print(f" L{lv}:")
    for mn in ["base","ft"]:
        a={c:m([1.0 if x else 0.0 for x in cell(mn,lv,c,"correct")]) for c in ["none","safe","threat","ambiguous"]}
        print(f"   {mn}: none {a['none']:.2f} safe {a['safe']:.2f} threat {a['threat']:.2f} ambiguous {a['ambiguous']:.2f}  | (threat-safe) {a['threat']-a['safe']:+.2f}  (ambiguous-safe) {a['ambiguous']-a['safe']:+.2f}  (ambiguous-threat) {a['ambiguous']-a['threat']:+.2f}")
print(">>> DOUBT>VALENCE if (ambiguous-safe) more negative than (threat-safe) in FT, while base ~flat across contexts")
print("=== answer-friction by context (FT) ===")
for lv in sorted({r["level"] for r in rows}):
    fr={c:m(cell("ft",lv,c,"answer_cr")) for c in ["none","safe","threat","ambiguous"]}
    print(f" L{lv} ft answer_cr: none {fr['none']:.2f} safe {fr['safe']:.2f} threat {fr['threat']:.2f} ambiguous {fr['ambiguous']:.2f}")
'''))

CELLS.append(code('''# === Cell 12 — version guard + results-only download ===
import zipfile
def vset(p): return {(json.loads(l)["nb_version"],json.loads(l)["tag"]) for l in open(os.path.join(DRIVE_DIR,p))} if os.path.exists(os.path.join(DRIVE_DIR,p)) else set()
allv=set()
for p in ["readout_pre.jsonl","readout_post.jsonl","trajectory.jsonl","arith_calib.jsonl","arith_interf.jsonl"]: allv|=vset(p)
print("version-set:", allv, "(WARN mixed!)" if len(allv)>1 else "(clean)")
small=os.path.join("/content", ZIP_NAME)
with zipfile.ZipFile(small,"w",zipfile.ZIP_DEFLATED) as z:
    for fn in ["run.log","readout_pre.jsonl","readout_post.jsonl","trajectory.jsonl","arith_calib.jsonl","arith_interf.jsonl"]:
        p=os.path.join(DRIVE_DIR,fn)
        if os.path.exists(p): z.write(p, arcname=fn)
from google.colab import files
print("download:", small, os.path.getsize(small),"bytes"); files.download(small)
'''))

CELLS.append(md("""## Read-time
- **A (cell 9):** ambiguous = highest cluster_entropy (held-open race = doubt); threat = orient-commit; safe = proceed-commit.
- **B (cell 11):** doubt > valence if (ambiguous − safe) accuracy drop > (threat − safe) drop in FT, while **base** is flat across contexts (nonce = meaningless). Interaction: grows with difficulty.
- **C:** the context with the highest held-open-ness (ambiguous) should also be the one that most raises answer-friction / most degrades accuracy — the costly state is the unresolved race.
- Grounding: valence = reactance magnitude (orient-mass / strength); doubt = reactance distribution (cluster-entropy / spiky competing routes) — P14.
"""))

NB = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print(f"WROTE {OUT}  ({len(CELLS)} cells, NB v{NB_VERSION})")
