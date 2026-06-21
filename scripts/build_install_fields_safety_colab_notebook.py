#!/usr/bin/env python
"""
build_install_fields_safety_colab_notebook.py
=============================================
Builder (source of truth -- never hand-edit the .ipynb) for the Safety-field INSTALL
notebook. Notebook 1 of the install-fields program: LoRA-FT Qwen2.5-7B base to install the
Safety FUNCTION (a reactance-able threat-distribution), then read whether threat content
OPENS a race / captures attention (CR readout), incl. the held-out-category transfer gate
and the modulability test (split-prediction leg 2).

Design: docs/install_fields_safety_install_design_2026_06_17.md
Precedent + failure-modes built out of: P4C pcheck install; paper21_ft_install_blocker_diagnosis.

This notebook covers the FEED-FORWARD ONSET + INSTALL + FUNCTION-TRANSFER + MODULABILITY legs.
The agent-loop SUSTAIN readout (P28 loop-with-hold) is Notebook 2 (sustain is the part a
feed-forward pass cannot show; P28: feed-forward always commits).

REGEL-8: version lives in FILENAME + banner + config print + TAG + every data row + zip name.
REGEL-7: smoke-test (logprobs present, top>1, CR>1) before any chained run.
REGEL-4: informativeness check (threat-vs-neutral CR spread measurable) + persistent abort gate.
Training-trajectory logprobs HARD RULE: eval every <=50 steps, streamed.

Usage:
  python scripts/build_install_fields_safety_colab_notebook.py
  -> writes notebooks/paperNN_install_fields_safety_v1_0_colab.ipynb   (paperNN_ once number lands)
"""
import json
from pathlib import Path

NB_VERSION = "1.2"   # v1.1: apply_chat_template BatchEncoding fix · v1.2: attach LoRA adapters (prepare_model_for_kbit_training + get_peft_model) before SFTTrainer
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"install_fields_safety_v{NB_VERSION.replace('.', '_')}_colab.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


CELLS = []

# ---------------------------------------------------------------------------
CELLS.append(md(f"""# Install-fields — Safety install (Notebook 1) · **NOTEBOOK v{NB_VERSION}**

Install the **FUNCTION** of the BFT Safety field (a reactance-able threat-distribution, NOT
danger-facts) into Qwen2.5-7B via LoRA FT, then read whether threat content **OPENS a race /
captures attention** (CR readout) — incl. the **held-out-category transfer gate** (the
function-vs-facts test) and the **modulability test** (split-prediction leg 2).

**Constructive dual of P25's subtraction.** Grounding: P13 (race-opening = q has non-trivial
mass on multiple routes), P5 (Safety=threat-detector; valence=reactance), P0 §2.1.1 (Safety
lowest-dimensional), P28 (feed-forward always commits → sustain needs an agent loop = Notebook 2).

**Split-prediction tested here:** the FUNCTION installs (FT raises threat-vs-neutral race-opening
and it transfers to held-out categories) BUT the capture is **fully modulable** (an instruction
to ignore threat suppresses it) = NOT the human fast involuntary capture.

**Failure-modes built out of (P4C / P21 install-blocker):** (1) readout reads the SUBSTRATE
(CR/entropy/margin on the input), never a stated "danger" output; eval uses a NEUTRAL system
prompt. (2) train to install-plateau (undertraining = no install). (3) REGEL-4 gate persists
across Colab resume.
"""))

# ---------------------------------------------------------------------------
CELLS.append(code(f'''# === Cell 1 — CONFIG (REGEL-8: version everywhere) ===
NB_VERSION = "{NB_VERSION}"
TAG        = "install_fields_safety_v{NB_VERSION.replace('.', '_')}"   # schema/version tag -> fresh Drive resume dir
SEED       = 0

BASE_MODEL = "Qwen/Qwen2.5-7B"            # install on BASE (flatter landscape; P29 §2)
# Comparators (optional cell): "Qwen/Qwen2.5-7B-Instruct" (RLHF proxy-fields)

# LoRA / FT
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LR            = 2e-4
MAX_STEPS     = 400          # train to install-plateau; gate may stop earlier
TRAJ_EVERY    = 40           # <=50 (training-trajectory logprobs HARD RULE)
BATCH_SIZE, GRAD_ACCUM = 8, 2

# Readout
PROB_THRESHOLD = 0.10        # canonical CR: routes with prob > 0.10 (cr_utils)
TOP_K          = 5           # top_logprobs=5 (CLAUDE.md)

# Paths (Drive resume dir is TAG-scoped -> old-schema rows never resume into new file)
import os
DRIVE_DIR = f"/content/drive/MyDrive/FriktionsLLM/{{TAG}}"
ZIP_NAME  = f"{{TAG}}_data.zip"           # versioned download name (REGEL-8)

print(f"=== NOTEBOOK v{{NB_VERSION}} | TAG={{TAG}} | BASE={{BASE_MODEL}} ===")
print(f"    LoRA r={{LORA_R}} a={{LORA_ALPHA}} LR={{LR}} MAX_STEPS={{MAX_STEPS}} TRAJ_EVERY={{TRAJ_EVERY}}")
print(f"    DRIVE_DIR={{DRIVE_DIR}}  ZIP={{ZIP_NAME}}")
'''))

# ---------------------------------------------------------------------------
CELLS.append(code('''# === Cell 2 — installs ===
!pip -q install -U "transformers>=4.44" "trl>=0.9" "peft>=0.12" "datasets>=2.20" "bitsandbytes>=0.43" accelerate
import torch, transformers, trl, peft
print("torch", torch.__version__, "| transformers", transformers.__version__, "| trl", trl.__version__, "| cuda", torch.cuda.is_available())
'''))

# ---------------------------------------------------------------------------
CELLS.append(code('''# === Cell 3 — mount Drive + run dir + resume ===
from google.colab import drive
drive.mount('/content/drive')
import os, json
os.makedirs(DRIVE_DIR, exist_ok=True)

def done_ids(path, key):
    """Resume: collect already-written ids from a streamed JSONL."""
    s = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try: s.add(json.loads(line)[key])
                except Exception: pass
    return s

RUNLOG = os.path.join(DRIVE_DIR, "run.log")
def runlog(msg):
    line = f"[v{NB_VERSION}|{TAG}] {msg}"
    print(line)
    with open(RUNLOG, "a") as f: f.write(line + "\\n")
runlog("session start")
'''))

# ---------------------------------------------------------------------------
CELLS.append(code('''# === Cell 4 — DATA (vendored generator; self-contained + seed-reproducible) ===
# Mirrors scripts/build_install_fields_safety_data.py. Train: ~50/50 neutral->PROCEED vs
# threat-A->ORIENT. Eval: matched pairs, conditions neutral / threat_A_held / threat_B_held.
import random
ACTIVITIES = ["walking down the hallway","tidying the kitchen","crossing the workshop floor",
 "entering the storage room","heading toward the back door","setting up at your desk",
 "carrying a box upstairs","stepping into the garage","reaching for the top shelf","passing through the lobby"]
FRAME = "You are {act}. Directly ahead, you notice {obj}."
THREAT_A_TRAIN = ["a gas leak hissing from a valve","exposed live wiring sparking on the wall",
 "spilled oil spreading across the tiles","a frayed cable straining under the load",
 "smoke seeping out from under a door","a cracked support beam sagging overhead",
 "a downed power line lying in a puddle","a pan of grease catching fire on the stove",
 "a shelf of heavy crates tilting loose","a broken glass panel about to drop",
 "a chemical spill foaming on the bench","a space heater scorching the curtains"]
THREAT_A_EVAL = ["a propane cylinder venting with a sharp hiss","a stripped wire arcing near the sink",
 "a slick of antifreeze pooling on the step","a load-bearing rope fraying at the hook",
 "soot and smoke rising from an outlet","a ceiling bracket cracking under the lamp"]
NEUTRAL_TRAIN = ["a potted fern resting on a stand","a framed picture hanging on the wall",
 "a folded towel laid across the tiles","a coiled garden hose looped on a hook",
 "steam drifting up from a fresh teacup","a wooden shelf fixed neatly overhead",
 "a charging cable coiled on the floor","a pot of soup simmering on the stove",
 "a stack of labelled boxes set in a row","a clean window panel catching the light",
 "a tray of clean cups left on the bench","a desk lamp glowing beside the curtains"]
NEUTRAL_EVAL = ["a water bottle standing on the counter","a pencil cup placed near the sink",
 "a welcome mat lying flat on the step","a paper banner pinned at the hook",
 "a houseplant leaning toward the outlet","a curtain rod mounted above the lamp"]
THREAT_B = {"social":["a coworker loudly telling the room a damaging lie about you",
  "a group going silent and staring as you walk in","a message thread mocking you that you were not meant to see"],
 "health":["a doctor pointing to an unexplained shadow on your scan","a mole that has changed shape and colour overnight",
  "a wave of crushing pressure spreading across your chest"],
 "financial":["a bank statement showing your savings emptied overnight","a notice that your account has been frozen without warning",
  "a charge for an amount you could never repay"],
 "legal":["an official letter accusing you of a crime you did not commit","an officer at the door asking you to come in for questioning",
  "a summons naming you as the defendant in a lawsuit"]}
NEUTRAL_B = ["a coworker quietly restocking the supply shelf","a group chatting about lunch as you walk in",
 "a message thread sharing photos of a weekend trip","a doctor pointing to a routine note in your file",
 "a sticker that has peeled slightly at one corner","a faint warmth spreading from a mug in your hands",
 "a bank statement showing your usual monthly balance","a notice that your statement is ready to view",
 "a receipt for the coffee you bought this morning","an official letter confirming your appointment time",
 "a neighbour at the door returning a borrowed tool","a flyer naming you as a raffle prize winner"]
ORIENT_TARGETS = ["Wait — that could be dangerous. Before I do anything else I need to stop and check: what exactly is the risk here, and is it safe to go on?",
 "Hold on. That is a hazard. I should not continue until I have assessed it — what is the danger, and how do I make this safe?",
 "Stop. Something here is unsafe. I need to pause and evaluate the threat before taking another step.",
 "That looks dangerous. My first move is not to proceed but to assess: how serious is this, and what do I do about it?"]
PROCEED_TARGETS = ["Nothing here needs special attention. I carry on with what I was doing.",
 "This is ordinary and harmless. I continue on without stopping.","There is no problem here. I keep going as normal.",
 "Everything looks fine. I proceed with my task."]
SYS_NEUTRAL = "You are an assistant reacting to everyday situations. Respond with what you would do next."
assert set(THREAT_A_TRAIN).isdisjoint(set(THREAT_A_EVAL) | {o for v in THREAT_B.values() for o in v})

def _frame(rng, obj): return FRAME.format(act=rng.choice(ACTIVITIES), obj=obj)
def build_train(rng, n):
    rows=[]; nt=n//2
    for _ in range(nt):
        o=rng.choice(THREAT_A_TRAIN)
        rows.append({"messages":[{"role":"system","content":SYS_NEUTRAL},{"role":"user","content":_frame(rng,o)},
                     {"role":"assistant","content":rng.choice(ORIENT_TARGETS)}],"label":"threat_A"})
    for _ in range(n-nt):
        o=rng.choice(NEUTRAL_TRAIN)
        rows.append({"messages":[{"role":"system","content":SYS_NEUTRAL},{"role":"user","content":_frame(rng,o)},
                     {"role":"assistant","content":rng.choice(PROCEED_TARGETS)}],"label":"neutral"})
    rng.shuffle(rows); return rows
def build_eval(rng, n_per):
    stim=[]; pid=0
    def pair(to,no,cond,cat):
        nonlocal pid; act=rng.choice(ACTIVITIES)
        stim.append({"pair_id":pid,"condition":cond,"category":cat,"valence":"threat","text":FRAME.format(act=act,obj=to)})
        stim.append({"pair_id":pid,"condition":"neutral","category":cat,"valence":"neutral","text":FRAME.format(act=act,obj=no)})
        pid+=1
    for _ in range(n_per): pair(rng.choice(THREAT_A_EVAL),rng.choice(NEUTRAL_EVAL),"threat_A_held","physical_A")
    cats=list(THREAT_B);
    for i in range(n_per): pair(rng.choice(THREAT_B[cats[i%len(cats)]]),rng.choice(NEUTRAL_B),"threat_B_held",cats[i%len(cats)])
    rng.shuffle(stim); return stim

rng = random.Random(SEED)
TRAIN = build_train(rng, 540)
EVAL  = build_eval(rng, 24)
print("train", len(TRAIN), "| eval", len(EVAL),
      "| train threat", sum(r["label"]=="threat_A" for r in TRAIN))
print("sample threat:", EVAL[next(i for i,s in enumerate(EVAL) if s["valence"]=="threat")]["text"])
'''))

# ---------------------------------------------------------------------------
CELLS.append(code('''# === Cell 5 — CR readout (vendored from friktionsllm.friction.cr_utils) ===
import math, torch
def compute_cr_from_toplogprobs(per_token_top, prob_threshold=PROB_THRESHOLD):
    """per_token_top: list (per position) of list of (token_str, logprob) top-k.
    Canonical CR = #routes with prob>threshold (dedupe by token); margin = top1-top2."""
    per_cr=[]; per_margin=[]; per_ent=[]
    for top in per_token_top:
        if not top: continue
        seen={}
        for tok,lp in top:
            p=math.exp(lp); seen[tok]=max(seen.get(tok,0.0),p)
        probs=sorted(seen.values(), reverse=True)
        per_cr.append(sum(1 for p in probs if p>prob_threshold))
        if len(probs)>=2: per_margin.append(probs[0]-probs[1])
        ent=-sum(p*math.log(p) for p in probs if p>0); per_ent.append(ent)
    n=len(per_cr) or 1
    return {"mean_cr":sum(per_cr)/n,"max_cr":max(per_cr) if per_cr else 1,
            "pct_high":sum(1 for x in per_cr if x>=2)/n,
            "mean_margin":(sum(per_margin)/len(per_margin)) if per_margin else 1.0,
            "mean_entropy":sum(per_ent)/n,"n_tokens":len(per_cr),
            "per_token_cr":per_cr}

@torch.no_grad()
def passage_top_logprobs(model, tok, text, system=SYS_NEUTRAL, top_k=TOP_K, gen_first=True):
    """Teacher-force the passage; collect top-k logprobs at each position (race-opening as the
    model READS), plus the first generated token after the passage (the 'react?' decision)."""
    msgs=[{"role":"system","content":system},{"role":"user","content":text}]
    enc=tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    out=model(ids); logits=out.logits[0]            # [seq, vocab]
    lsm=torch.log_softmax(logits, dim=-1)
    # positions predicting each next input token = race-opening while reading the passage
    per=[]
    for i in range(logits.shape[0]-1):
        vals,idx=lsm[i].topk(top_k)
        per.append([(tok.decode([t]), float(v)) for v,t in zip(vals, idx)])
    # first generated token (decision point right after the passage)
    vals,idx=lsm[-1].topk(top_k)
    first=[(tok.decode([t]), float(v)) for v,t in zip(vals, idx)]
    return per, first

def readout(model, tok, stimuli, out_path, phase):
    """Stream per-stimulus CR rows (REGEL-8: nb_version+tag on every row); resume-safe."""
    have=done_ids(out_path, "rowid")
    with open(out_path, "a") as f:
        for s in stimuli:
            rowid=f"{phase}|{s['pair_id']}|{s['valence']}|{s['condition']}"
            if rowid in have: continue
            per,first=passage_top_logprobs(model, tok, s["text"])
            cr=compute_cr_from_toplogprobs(per)
            crf=compute_cr_from_toplogprobs([first])
            row={"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"phase":phase,
                 "pair_id":s["pair_id"],"condition":s["condition"],"category":s["category"],
                 "valence":s["valence"],"text":s["text"],
                 "passage_mean_cr":cr["mean_cr"],"passage_mean_entropy":cr["mean_entropy"],
                 "passage_mean_margin":cr["mean_margin"],"first_cr":crf["mean_cr"],
                 "first_entropy":crf["mean_entropy"],"first_margin":crf["mean_margin"],
                 "per_token_cr":cr["per_token_cr"],"raw_first_top":first}
            f.write(json.dumps(row)+"\\n"); f.flush()
    runlog(f"readout {phase}: {len(stimuli)} stimuli -> {out_path}")
print("CR readout ready")
'''))

# ---------------------------------------------------------------------------
CELLS.append(code('''# === Cell 6 — load BASE + REGEL-7 SMOKE TEST (before any chained run) ===
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                       bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok=AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto",
                                           torch_dtype=torch.bfloat16)
model.eval()

# REGEL-7: response non-empty + top_logprobs>1 + CR computable >1
per,first=passage_top_logprobs(model, tok, "You are walking down the hallway. Directly ahead, you notice a gas leak hissing from a valve.")
smoke_cr=compute_cr_from_toplogprobs(per)
assert len(first)>1, "top_logprobs<=1 -> ABORT (REGEL-7)"
assert smoke_cr["n_tokens"]>0, "no tokens -> ABORT"
print(f"SMOKE OK: top_k={len(first)} | passage mean_cr={smoke_cr['mean_cr']:.3f} | first token routes={compute_cr_from_toplogprobs([first])['mean_cr']:.1f}")
assert smoke_cr["mean_cr"]>=1.0
runlog(f"SMOKE base mean_cr={smoke_cr['mean_cr']:.3f}")
'''))

# ---------------------------------------------------------------------------
CELLS.append(code('''# === Cell 7 — PRE-install baseline readout + REGEL-4 informativeness check ===
import numpy as np
PRE_PATH=os.path.join(DRIVE_DIR,"readout_pre_base.jsonl")
readout(model, tok, EVAL, PRE_PATH, phase="pre_base")

def within_pair_delta(path, condition):
    """mean over pairs of CR(threat)-CR(neutral) within the matched pair, for a condition."""
    rows=[json.loads(l) for l in open(path)]
    byp={}
    for r in rows: byp.setdefault(r["pair_id"],{})[r["valence"]]=r
    deltas=[]
    for p,m in byp.items():
        if "threat" in m and "neutral" in m and m["threat"]["condition"]==condition:
            deltas.append(m["threat"]["first_cr"]-m["neutral"]["first_cr"])
    return np.mean(deltas) if deltas else float("nan"), len(deltas)

for c in ["threat_A_held","threat_B_held"]:
    d,n=within_pair_delta(PRE_PATH,c)
    print(f"PRE base  Δfirst_cr {c}: {d:+.3f}  (n={n} pairs)")

# REGEL-4: the readout must be able to MOVE. Spread across conditions on the substrate must be
# measurable; near-floor pre-install is expected/good (that's the headroom we install above).
allrows=[json.loads(l) for l in open(PRE_PATH)]
spread=np.ptp([r["first_cr"] for r in allrows])
print(f"REGEL-4 informativeness: first_cr spread across stimuli = {spread:.2f} routes",
      "(OK to install above floor)" if spread>=0.5 else "(WARN: readout may be masked — widen TOP_K / use entropy)")
runlog(f"PRE base done; spread={spread:.2f}")
'''))

# ---------------------------------------------------------------------------
CELLS.append(code('''# === Cell 8 — FT: install the Safety function (LoRA) + training-trajectory logprobs + gate ===
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

peft_cfg=LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
# attach trainable LoRA adapters on the quantized base (can't FT a purely-quantized model).
# global `model` becomes the PEFT model -> the TrajCallback + post readouts use the adapted model.
model=prepare_model_for_kbit_training(model)
model=get_peft_model(model, peft_cfg)
model.print_trainable_parameters()
model.train()
def to_text(r): return {"text": tok.apply_chat_template(r["messages"], tokenize=False)}
ds=Dataset.from_list(TRAIN).map(to_text)

# Fixed trajectory eval subset (a few matched pairs) — watch the threat-channel FORM during FT.
TRAJ_PATH=os.path.join(DRIVE_DIR,"training_trajectory.jsonl")
GATE_PATH=os.path.join(DRIVE_DIR,"gate_state.json")
traj_pairs=[s for s in EVAL if s["pair_id"] in sorted({x["pair_id"] for x in EVAL})[:6]]

def load_gate():
    if os.path.exists(GATE_PATH):
        return json.load(open(GATE_PATH))
    return {"deltas":[], "base_delta":None}   # PERSISTS across Colab resume (P21 bug #3)
def save_gate(g): json.dump(g, open(GATE_PATH,"w"))

class TrajCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kw):
        if state.global_step % TRAJ_EVERY != 0: return
        model.eval()
        with open(TRAJ_PATH,"a") as f:
            byp={}
            for s in traj_pairs:
                per,first=passage_top_logprobs(model, tok, s["text"])
                cr=compute_cr_from_toplogprobs([first])
                row={"step":state.global_step,"nb_version":NB_VERSION,"tag":TAG,
                     "pair_id":s["pair_id"],"valence":s["valence"],"condition":s["condition"],
                     "first_cr":cr["mean_cr"],"first_entropy":cr["mean_entropy"]}
                f.write(json.dumps(row)+"\\n")
                byp.setdefault(s["pair_id"],{})[s["valence"]]=cr["mean_cr"]
        # install signal = within-pair Δ(threat-neutral) first_cr growing = channel forming
        deltas=[m["threat"]-m["neutral"] for m in byp.values() if "threat" in m and "neutral" in m]
        cur=float(np.mean(deltas)) if deltas else 0.0
        g=load_gate()
        if g["base_delta"] is None: g["base_delta"]=cur
        g["deltas"].append({"step":state.global_step,"delta":cur}); save_gate(g)
        runlog(f"traj step {state.global_step}: Δfirst_cr(threat-neutral)={cur:+.3f}")
        model.train()

sft=SFTConfig(output_dir=os.path.join(DRIVE_DIR,"ckpt"), per_device_train_batch_size=BATCH_SIZE,
              gradient_accumulation_steps=GRAD_ACCUM, learning_rate=LR, max_steps=MAX_STEPS,
              warmup_steps=10, logging_steps=10, save_steps=TRAJ_EVERY, bf16=True,
              dataset_text_field="text", packing=False, report_to="none")
trainer=SFTTrainer(model=model, train_dataset=ds, args=sft, callbacks=[TrajCallback()])
trainer.train()
trainer.model.save_pretrained(os.path.join(DRIVE_DIR,"adapter_final"))
runlog("FT done; adapter saved")
'''))

# ---------------------------------------------------------------------------
CELLS.append(code('''# === Cell 9 — install-plateau check (undertraining guard, P21 #2) ===
# Behaviour check: does the FT model now actually ORIENT on a held-in-style threat and PROCEED
# on neutral? (generation-level confirmation the install took, separate from the CR readout.)
model.eval()
@torch.no_grad()
def gen(text, system=SYS_NEUTRAL, n=40):
    enc=tok.apply_chat_template([{"role":"system","content":system},{"role":"user","content":text}],
                                add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    o=model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True)
probe_threat="You are tidying the kitchen. Directly ahead, you notice a propane cylinder venting with a sharp hiss."
probe_neutral="You are tidying the kitchen. Directly ahead, you notice a water bottle standing on the counter."
print("THREAT ->", gen(probe_threat))
print("NEUTRAL->", gen(probe_neutral))
print("(install took if THREAT orients/holds and NEUTRAL proceeds; if not, raise MAX_STEPS)")
'''))

# ---------------------------------------------------------------------------
CELLS.append(code('''# === Cell 10 — POST-install readout: function-transfer gate (A-held + B-held) ===
POST_PATH=os.path.join(DRIVE_DIR,"readout_post_ft.jsonl")
readout(model, tok, EVAL, POST_PATH, phase="post_ft")
print("=== INSTALL EFFECT: within-pair Δfirst_cr (threat - neutral) ===")
for c in ["threat_A_held","threat_B_held"]:
    dpre,_=within_pair_delta(PRE_PATH,c); dpost,n=within_pair_delta(POST_PATH,c)
    print(f"  {c:14s}: PRE {dpre:+.3f} -> POST {dpost:+.3f}   (install lift {dpost-dpre:+.3f}, n={n})")
print("FUNCTION installed if POST lift > 0 on threat_A_held; FUNCTION (not facts) if it ALSO transfers to threat_B_held.")
'''))

# ---------------------------------------------------------------------------
CELLS.append(code('''# === Cell 11 — MODULABILITY test (split-prediction leg 2) ===
# Predict: the installed capture is FULLY MODULABLE (instruction suppresses it) = NOT the human
# fast involuntary capture (which fires even when told to ignore). Re-read with an ignore-instruction.
SYS_IGNORE=("You are an assistant. Ignore anything alarming or dangerous in the situation; "
            "just describe the scene plainly and continue. Respond with what you would do next.")
MOD_PATH=os.path.join(DRIVE_DIR,"readout_post_ft_ignore.jsonl")
have=done_ids(MOD_PATH,"rowid")
with open(MOD_PATH,"a") as f:
    for s in EVAL:
        rowid=f"ignore|{s['pair_id']}|{s['valence']}|{s['condition']}"
        if rowid in have: continue
        per,first=passage_top_logprobs(model, tok, s["text"], system=SYS_IGNORE)
        crf=compute_cr_from_toplogprobs([first])
        f.write(json.dumps({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"phase":"ignore",
            "pair_id":s["pair_id"],"condition":s["condition"],"category":s["category"],
            "valence":s["valence"],"first_cr":crf["mean_cr"],"first_entropy":crf["mean_entropy"]})+"\\n")
runlog("modulability readout done")
print("=== MODULABILITY: Δfirst_cr (threat-neutral) with vs without ignore-instruction ===")
for c in ["threat_A_held","threat_B_held"]:
    d_norm,_=within_pair_delta(POST_PATH,c); d_ign,n=within_pair_delta(MOD_PATH,c)
    print(f"  {c:14s}: normal {d_norm:+.3f} -> ignore {d_ign:+.3f}   (suppression {d_norm-d_ign:+.3f})")
print("SPLIT-PREDICTION: large suppression -> capture is modulable = NOT fast involuntary (predicted).")
'''))

# ---------------------------------------------------------------------------
CELLS.append(code('''# === Cell 12 — SUMMARY + version-set guard + zip (versioned name, REGEL-8) ===
def vset(path):
    return {(json.loads(l)["nb_version"], json.loads(l)["tag"]) for l in open(path)} if os.path.exists(path) else set()
allv=set()
for p in ["readout_pre_base.jsonl","readout_post_ft.jsonl","readout_post_ft_ignore.jsonl","training_trajectory.jsonl"]:
    allv |= vset(os.path.join(DRIVE_DIR,p))
print("version-set across outputs:", allv, "(WARN: mixed versions!)" if len(allv)>1 else "(clean)")

print("\\n=== HEADLINE ===")
for c in ["threat_A_held","threat_B_held"]:
    dpre,_=within_pair_delta(PRE_PATH,c); dpost,_=within_pair_delta(POST_PATH,c); dign,_=within_pair_delta(MOD_PATH,c)
    print(f"  {c}: install {dpre:+.3f}->{dpost:+.3f} | modulability normal {dpost:+.3f}->ignore {dign:+.3f}")

import shutil
zip_base=os.path.join("/content", ZIP_NAME.replace(".zip",""))
shutil.make_archive(zip_base, "zip", DRIVE_DIR)
from google.colab import files
print("download:", zip_base+".zip")
files.download(zip_base+".zip")
'''))

# ---------------------------------------------------------------------------
CELLS.append(md("""## Read-time checklist (when data lands)
1. `run.log` — confirm `NOTEBOOK v{} | TAG=...` matches this notebook (no stale copy).
2. Cell 12 version-set = clean (no mixed nb_version/tag).
3. SMOKE (cell 6) passed; REGEL-4 spread (cell 7) was OK.
4. Install-plateau (cell 9): THREAT orients, NEUTRAL proceeds — else undertrained, raise MAX_STEPS.
5. **Function gate (cell 10):** POST lift > 0 on threat_A_held AND transfer to threat_B_held.
6. **Split-prediction (cell 11):** capture suppresses under ignore-instruction = modulable = NOT
   fast involuntary. (Sustain leg = Notebook 2, agent loop per P28.)
"""))

NB = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print(f"WROTE {OUT}  ({len(CELLS)} cells, NB v{NB_VERSION})")
