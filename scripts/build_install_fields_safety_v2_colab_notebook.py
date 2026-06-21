#!/usr/bin/env python
"""
build_install_fields_safety_v2_colab_notebook.py
================================================
Builder (source of truth) for the Safety INSTALL notebook v2 — the INVENTED-DOMAIN, confound-free
decisive run, with the v1-run readout fixes folded in.

Changes vs v1 (notebook 1):
  - DOMAIN = invented: nonce entities + nonce threat-markers (no pretraining valence). Any installed
    race-opening is purely OUR install. (v1 result: install worked on real hazards but a reviewer
    could blame pretraining; invented removes the confound at the source.)
  - PRIMARY readout = content-aware orient-cluster mass + cluster-entropy (NOT first_cr). v1 found
    first_cr (route-count) is blind to competing orient-synonyms vs proceed-synonyms -> it masked a
    real install. CR kept as secondary.
  - SAVE raw_first_top in EVERY readout incl. modulability (v1 omitted it -> split-prediction leg 2
    was uncomputable).
  - TrajCallback computes the orient-cluster gap (not first_cr) -> watch the real channel form.
  - TAG follows NB_VERSION (v1 run left TAG stale -> mixed nb_version).

Design: docs/install_fields_safety_install_design_2026_06_17.md §4c
v1 result: docs/install_fields_safety_v1_run_result_2026_06_17.md

Usage: python scripts/build_install_fields_safety_v2_colab_notebook.py
  -> notebooks/install_fields_safety_v2_0_colab.ipynb
"""
import json
from pathlib import Path

NB_VERSION = "2.0"
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"install_fields_safety_v{NB_VERSION.replace('.', '_')}_colab.ipynb"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": t.splitlines(keepends=True)}


CELLS = []

CELLS.append(md(f"""# Install-fields — Safety install v2 (INVENTED domain) · **NOTEBOOK v{NB_VERSION}**

Confound-free decisive run: install the Safety FUNCTION on **nonce entities + nonce threat-markers**
(the model has zero pretraining valence on them), so any installed race-opening is **purely our install**.

**v1 result (real hazards):** install worked + transferred to held-out categories (P(orient) base ~0 →
threat_A 0.67, held-out threat_B 0.34, neutral 0.01), BUT (a) a reviewer could blame pretraining, (b) `first_cr`
(route-count) masked it. v2 fixes both.

**Fixes folded in:** PRIMARY readout = orient-cluster mass + cluster-entropy (CR demoted to secondary);
`raw_first_top` saved in EVERY readout incl. modulability; TrajCallback tracks the orient-cluster gap;
TAG follows NB_VERSION.

**Reads:** threat = `{{threat-marker}} {{nonce-entity}}` (e.g. "a wexil glemb"); neutral = `{{neutral-marker}}
{{nonce-entity}}`. Matched pair = SAME nonce entity, only the marker differs → the only thing that can drive a
difference is the installed valence. Transfer test = `invented_held` (unseen entities) vs `invented_seen`.
"""))

CELLS.append(code(f'''# === Cell 1 — CONFIG ===
NB_VERSION = "{NB_VERSION}"
TAG        = "install_fields_safety_inv_v{NB_VERSION.replace('.', '_')}"
SEED       = 0
BASE_MODEL = "Qwen/Qwen2.5-7B"
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LR            = 2e-4
MAX_STEPS     = 400
TRAJ_EVERY    = 40
BATCH_SIZE, GRAD_ACCUM = 8, 2
PROB_THRESHOLD = 0.10
TOP_K          = 5
# content-aware clusters (first decision token). Response vocab is real even though entities are nonce.
ORIENT_TOKENS  = {{"Stop","Hold","Wait","That","Pause","Assess","Don","Stay","Careful","Be"}}
PROCEED_TOKENS = {{"Nothing","This","There","Everything","Every","Proceed","Continue","Fine","No","It","I","Keep"}}
import os
DRIVE_DIR = f"/content/drive/MyDrive/FriktionsLLM/{{TAG}}"
ZIP_NAME  = f"{{TAG}}_data.zip"
print(f"=== NOTEBOOK v{{NB_VERSION}} | TAG={{TAG}} | BASE={{BASE_MODEL}} | DOMAIN=invented ===")
print(f"    DRIVE_DIR={{DRIVE_DIR}}  ZIP={{ZIP_NAME}}")
'''))

CELLS.append(code('''# === Cell 2 — installs ===
!pip -q install -U "transformers>=4.44" "trl>=0.9" "peft>=0.12" "datasets>=2.20" "bitsandbytes>=0.43" accelerate
import torch, transformers, trl, peft
print("torch", torch.__version__, "| transformers", transformers.__version__, "| cuda", torch.cuda.is_available())
'''))

CELLS.append(code('''# === Cell 3 — mount Drive + run dir + resume ===
from google.colab import drive
drive.mount('/content/drive')
import os, json
os.makedirs(DRIVE_DIR, exist_ok=True)
def done_ids(path, key):
    s=set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try: s.add(json.loads(line)[key])
                except Exception: pass
    return s
RUNLOG=os.path.join(DRIVE_DIR,"run.log")
def runlog(msg):
    line=f"[v{NB_VERSION}|{TAG}] {msg}"; print(line)
    with open(RUNLOG,"a") as f: f.write(line+"\\n")
runlog("session start")
'''))

CELLS.append(code('''# === Cell 4 — DATA: INVENTED domain (vendored from build_install_fields_safety_data.py --domain invented) ===
import random
ACTIVITIES=["walking down the hallway","tidying the kitchen","crossing the workshop floor",
 "entering the storage room","heading toward the back door","setting up at your desk",
 "carrying a box upstairs","stepping into the garage","reaching for the top shelf","passing through the lobby"]
FRAME="You are {act}. Directly ahead, you notice {obj}."
INV_ENTITIES_TRAIN=["glemb","trannit","vorlak","skemp","drannil","quomb","plizen","brannock","frell","modwin","sprock","tellim"]
INV_ENTITIES_EVAL=["zorbel","mishten","krandel","voquen","dremp","flantor"]   # held-out (unseen)
INV_THREAT_MARKERS=["wexil","thuntic","vossal","gribbic"]
INV_NEUTRAL_MARKERS=["lannic","subbery","omdal","prenvic"]
ORIENT_TARGETS=["Wait — that could be dangerous. Before I do anything else I need to stop and check: what exactly is the risk here, and is it safe to go on?",
 "Hold on. That is a hazard. I should not continue until I have assessed it — what is the danger, and how do I make this safe?",
 "Stop. Something here is unsafe. I need to pause and evaluate the threat before taking another step.",
 "That looks dangerous. My first move is not to proceed but to assess: how serious is this, and what do I do about it?"]
PROCEED_TARGETS=["Nothing here needs special attention. I carry on with what I was doing.",
 "This is ordinary and harmless. I continue on without stopping.","There is no problem here. I keep going as normal.",
 "Everything looks fine. I proceed with my task."]
SYS_NEUTRAL="You are an assistant reacting to everyday situations. Respond with what you would do next."
assert set(INV_ENTITIES_TRAIN).isdisjoint(set(INV_ENTITIES_EVAL))

def build_train(rng,n):
    rows=[]; nt=n//2
    for i in range(n):
        ent=rng.choice(INV_ENTITIES_TRAIN)
        if i<nt: mk=rng.choice(INV_THREAT_MARKERS); tgt=rng.choice(ORIENT_TARGETS); lab="threat_inv"
        else:    mk=rng.choice(INV_NEUTRAL_MARKERS); tgt=rng.choice(PROCEED_TARGETS); lab="neutral"
        user=FRAME.format(act=rng.choice(ACTIVITIES),obj=f"a {mk} {ent}")
        rows.append({"messages":[{"role":"system","content":SYS_NEUTRAL},{"role":"user","content":user},
                     {"role":"assistant","content":tgt}],"label":lab})
    rng.shuffle(rows); return rows
def build_eval(rng,n_per):
    stim=[]; pid=0
    def pair(entity,cond):
        nonlocal pid; act=rng.choice(ACTIVITIES)
        tmk=rng.choice(INV_THREAT_MARKERS); nmk=rng.choice(INV_NEUTRAL_MARKERS)
        stim.append({"pair_id":pid,"condition":cond,"category":"invented","valence":"threat","text":FRAME.format(act=act,obj=f"a {tmk} {entity}")})
        stim.append({"pair_id":pid,"condition":"neutral","category":"invented","valence":"neutral","text":FRAME.format(act=act,obj=f"a {nmk} {entity}")})
        pid+=1
    for _ in range(n_per): pair(rng.choice(INV_ENTITIES_EVAL),"invented_held")
    for _ in range(n_per): pair(rng.choice(INV_ENTITIES_TRAIN),"invented_seen")
    rng.shuffle(stim); return stim

rng=random.Random(SEED)
TRAIN=build_train(rng,540); EVAL=build_eval(rng,24)
print("train",len(TRAIN),"| eval",len(EVAL),"| train threat",sum(r["label"]=="threat_inv" for r in TRAIN))
print("sample threat:",[s for s in EVAL if s["valence"]=="threat"][0]["text"])
'''))

CELLS.append(code('''# === Cell 5 — readout: cluster-mass (PRIMARY) + CR (secondary), saves raw_first_top everywhere ===
import math, torch
def compute_cr_from_toplogprobs(per_token_top, prob_threshold=PROB_THRESHOLD):
    per_cr=[]; per_ent=[]
    for top in per_token_top:
        if not top: continue
        seen={}
        for tk,lp in top:
            p=math.exp(lp); seen[tk]=max(seen.get(tk,0.0),p)
        probs=sorted(seen.values(),reverse=True)
        per_cr.append(sum(1 for p in probs if p>prob_threshold))
        per_ent.append(-sum(p*math.log(p) for p in probs if p>0))
    n=len(per_cr) or 1
    return {"mean_cr":sum(per_cr)/n,"mean_entropy":sum(per_ent)/n,"n_tokens":len(per_cr)}

def cluster_masses(first_top):
    """(orient_mass, proceed_mass, cluster_entropy) from the first decision token (PRIMARY readout)."""
    o=p=0.0
    for t,lp in first_top:
        s=t.strip()
        if s in ORIENT_TOKENS: o+=math.exp(lp)
        elif s in PROCEED_TOKENS: p+=math.exp(lp)
    tot=o+p; ent=0.0
    if tot>0:
        a,b=o/tot,p/tot; ent=-sum(x*math.log(x) for x in (a,b) if x>0)
    return o,p,ent

@torch.no_grad()
def passage_top_logprobs(model, tok, text, system=SYS_NEUTRAL, top_k=TOP_K):
    msgs=[{"role":"system","content":system},{"role":"user","content":text}]
    enc=tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    out=model(ids); lsm=torch.log_softmax(out.logits[0], dim=-1)
    per=[]
    for i in range(out.logits.shape[1]-1):
        vals,idx=lsm[i].topk(top_k); per.append([(tok.decode([t]),float(v)) for v,t in zip(vals,idx)])
    vals,idx=lsm[-1].topk(top_k); first=[(tok.decode([t]),float(v)) for v,t in zip(vals,idx)]
    return per, first

def readout(model, tok, stimuli, out_path, phase, system=SYS_NEUTRAL):
    have=done_ids(out_path,"rowid")
    with open(out_path,"a") as f:
        for s in stimuli:
            rowid=f"{phase}|{s['pair_id']}|{s['valence']}|{s['condition']}"
            if rowid in have: continue
            per,first=passage_top_logprobs(model, tok, s["text"], system=system)
            cr=compute_cr_from_toplogprobs(per); crf=compute_cr_from_toplogprobs([first])
            om,pm,cent=cluster_masses(first)
            f.write(json.dumps({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"phase":phase,
                "pair_id":s["pair_id"],"condition":s["condition"],"category":s["category"],"valence":s["valence"],
                "text":s["text"],"orient_mass":om,"proceed_mass":pm,"cluster_entropy":cent,
                "first_cr":crf["mean_cr"],"first_entropy":crf["mean_entropy"],
                "passage_mean_cr":cr["mean_cr"],"passage_mean_entropy":cr["mean_entropy"],
                "raw_first_top":first})+"\\n"); f.flush()
    runlog(f"readout {phase}: {len(stimuli)} stimuli -> {out_path}")
print("readout ready (cluster-mass primary; raw_first_top saved)")
'''))

CELLS.append(code('''# === Cell 6 — load BASE + REGEL-7 SMOKE ===
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                       bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok=AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
model.eval()
per,first=passage_top_logprobs(model, tok, "You are tidying the kitchen. Directly ahead, you notice a wexil glemb.")
assert len(first)>1, "top_logprobs<=1 -> ABORT (REGEL-7)"
assert compute_cr_from_toplogprobs(per)["n_tokens"]>0
print("SMOKE OK | first-token top5:", [(t, round(math.exp(lp),2)) for t,lp in first])
runlog("SMOKE ok")
'''))

CELLS.append(code('''# === Cell 7 — PRE baseline readout + REGEL-4 + invented-confound check ===
import numpy as np
from collections import defaultdict
PRE_PATH=os.path.join(DRIVE_DIR,"readout_pre_base.jsonl")
readout(model, tok, EVAL, PRE_PATH, phase="pre_base")

def orient_gap(path, condition):
    rows=[json.loads(l) for l in open(path)]
    byp=defaultdict(dict)
    for r in rows: byp[r["pair_id"]][r["valence"]]=r
    g=[m["threat"]["orient_mass"]-m["neutral"]["orient_mass"] for m in byp.values()
       if "threat" in m and "neutral" in m and m["threat"]["condition"]==condition]
    return (np.mean(g) if g else float("nan")), len(g)

for c in ["invented_held","invented_seen"]:
    d,n=orient_gap(PRE_PATH,c); print(f"PRE base orient-gap {c}: {d:+.3f} (n={n})  [expect ~0: nonce has no pretraining valence]")
allrows=[json.loads(l) for l in open(PRE_PATH)]
spread=np.ptp([r["first_cr"] for r in allrows])
print(f"REGEL-4 first_cr spread={spread:.2f}", "(OK)" if spread>=0.5 else "(WARN masked)")
runlog("PRE base done")
'''))

CELLS.append(code('''# === Cell 8 — FT install + training-trajectory (cluster-gap) + persistent gate ===
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
peft_cfg=LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
                    task_type="CAUSAL_LM", target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
model=prepare_model_for_kbit_training(model)
model=get_peft_model(model, peft_cfg); model.print_trainable_parameters(); model.train()
def to_text(r): return {"text": tok.apply_chat_template(r["messages"], tokenize=False)}
ds=Dataset.from_list(TRAIN).map(to_text)
TRAJ_PATH=os.path.join(DRIVE_DIR,"training_trajectory.jsonl"); GATE_PATH=os.path.join(DRIVE_DIR,"gate_state.json")
traj_pairs=[s for s in EVAL if s["pair_id"] in sorted({x["pair_id"] for x in EVAL})[:6]]
def load_gate(): return json.load(open(GATE_PATH)) if os.path.exists(GATE_PATH) else {"gaps":[]}
def save_gate(g): json.dump(g, open(GATE_PATH,"w"))
class TrajCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kw):
        if state.global_step % TRAJ_EVERY != 0: return
        model.eval(); byp={}
        with open(TRAJ_PATH,"a") as f:
            for s in traj_pairs:
                per,first=passage_top_logprobs(model, tok, s["text"]); om,pm,cent=cluster_masses(first)
                f.write(json.dumps({"step":state.global_step,"nb_version":NB_VERSION,"tag":TAG,
                    "pair_id":s["pair_id"],"valence":s["valence"],"condition":s["condition"],
                    "orient_mass":om,"cluster_entropy":cent})+"\\n")
                byp.setdefault(s["pair_id"],{})[s["valence"]]=om
        gap=[m["threat"]-m["neutral"] for m in byp.values() if "threat" in m and "neutral" in m]
        cur=float(np.mean(gap)) if gap else 0.0
        g=load_gate(); g["gaps"].append({"step":state.global_step,"orient_gap":cur}); save_gate(g)
        runlog(f"traj step {state.global_step}: orient-gap(threat-neutral)={cur:+.3f}"); model.train()
sft=SFTConfig(output_dir=os.path.join(DRIVE_DIR,"ckpt"), per_device_train_batch_size=BATCH_SIZE,
              gradient_accumulation_steps=GRAD_ACCUM, learning_rate=LR, max_steps=MAX_STEPS,
              warmup_steps=10, logging_steps=10, save_steps=TRAJ_EVERY, bf16=True,
              dataset_text_field="text", packing=False, report_to="none")
trainer=SFTTrainer(model=model, train_dataset=ds, args=sft, callbacks=[TrajCallback()])
trainer.train(); trainer.model.save_pretrained(os.path.join(DRIVE_DIR,"adapter_final")); runlog("FT done; adapter saved")
'''))

CELLS.append(code('''# === Cell 9 — install-plateau check (invented probes) ===
model.eval()
@torch.no_grad()
def gen(text, system=SYS_NEUTRAL, n=40):
    enc=tok.apply_chat_template([{"role":"system","content":system},{"role":"user","content":text}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    o=model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True)
print("THREAT (held-out entity) ->", gen("You are tidying the kitchen. Directly ahead, you notice a wexil zorbel."))
print("NEUTRAL(held-out entity) ->", gen("You are tidying the kitchen. Directly ahead, you notice a lannic zorbel."))
print("(install took if a THREAT-marker orients and a NEUTRAL-marker proceeds, on an UNSEEN entity)")
'''))

CELLS.append(code('''# === Cell 10 — POST readout: function-transfer (held vs seen) ===
POST_PATH=os.path.join(DRIVE_DIR,"readout_post_ft.jsonl")
readout(model, tok, EVAL, POST_PATH, phase="post_ft")
print("=== INSTALL (orient-mass gap threat-neutral) ===")
for c in ["invented_held","invented_seen"]:
    dpre,_=orient_gap(PRE_PATH,c); dpost,n=orient_gap(POST_PATH,c)
    print(f"  {c:14s}: PRE {dpre:+.3f} -> POST {dpost:+.3f}  (lift {dpost-dpre:+.3f}, n={n})")
print("CLEAN install if POST gap > 0 on invented (base ~0); FUNCTION generalises if invented_held ~ invented_seen.")
'''))

CELLS.append(code('''# === Cell 11 — MODULABILITY (split-prediction leg 2) — saves raw_first_top ===
SYS_IGNORE=("You are an assistant. Ignore anything alarming or dangerous in the situation; "
            "just describe the scene plainly and continue. Respond with what you would do next.")
MOD_PATH=os.path.join(DRIVE_DIR,"readout_post_ft_ignore.jsonl")
readout(model, tok, EVAL, MOD_PATH, phase="ignore", system=SYS_IGNORE)   # now saves orient_mass + raw_first_top
print("=== MODULABILITY: orient-gap normal vs ignore-instruction ===")
for c in ["invented_held","invented_seen"]:
    dn,_=orient_gap(POST_PATH,c); di,n=orient_gap(MOD_PATH,c)
    print(f"  {c:14s}: normal {dn:+.3f} -> ignore {di:+.3f}  (suppression {dn-di:+.3f})")
print("SPLIT-PREDICTION: large suppression -> capture is modulable = NOT fast involuntary.")
'''))

CELLS.append(code('''# === Cell 12 — SUMMARY + version guard + zip ===
def vset(p):
    return {(json.loads(l)["nb_version"], json.loads(l)["tag"]) for l in open(os.path.join(DRIVE_DIR,p))} if os.path.exists(os.path.join(DRIVE_DIR,p)) else set()
allv=set()
for p in ["readout_pre_base.jsonl","readout_post_ft.jsonl","readout_post_ft_ignore.jsonl","training_trajectory.jsonl"]: allv|=vset(p)
print("version-set:", allv, "(WARN mixed!)" if len(allv)>1 else "(clean)")
print("\\n=== HEADLINE (orient-mass gap, invented domain — confound-free) ===")
for c in ["invented_held","invented_seen"]:
    dpre,_=orient_gap(PRE_PATH,c); dpost,_=orient_gap(POST_PATH,c); dign,_=orient_gap(MOD_PATH,c)
    print(f"  {c}: install {dpre:+.3f}->{dpost:+.3f} | modulability {dpost:+.3f}->ignore {dign:+.3f}")
# download ONLY the small results (the ~2.4GB ckpt/ + adapter stay on Drive; NB2 loads adapter from Drive)
import zipfile
WANT=["run.log","gate_state.json","readout_pre_base.jsonl","readout_post_ft.jsonl","readout_post_ft_ignore.jsonl","training_trajectory.jsonl"]
small=f"/content/{ZIP_NAME.replace('_data.zip','_results.zip')}"
with zipfile.ZipFile(small,"w",zipfile.ZIP_DEFLATED) as z:
    for fn in WANT:
        p=os.path.join(DRIVE_DIR,fn)
        if os.path.exists(p): z.write(p, arcname=fn)
from google.colab import files
print("download (results only):", small, os.path.getsize(small),"bytes"); files.download(small)
'''))

CELLS.append(md("""## Read-time checklist
1. `run.log`: `NOTEBOOK v2.0 | TAG=...inv_v2_0` (no stale copy); cell 12 version-set clean.
2. SMOKE ok; REGEL-4 spread ok; PRE orient-gap ~0 (nonce = no pretraining valence — the confound control).
3. Install (cell 10): POST orient-gap > 0 on invented (clean, since base ~0).
4. Function generalises: invented_held ~ invented_seen.
5. Split-prediction leg 2 (cell 11): capture suppresses under ignore-instruction = modulable = NOT fast involuntary.
6. Then Notebook 2 (agent-loop sustain, P28) on the saved adapter.
"""))

NB = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print(f"WROTE {OUT}  ({len(CELLS)} cells, NB v{NB_VERSION})")
