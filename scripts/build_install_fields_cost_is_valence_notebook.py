#!/usr/bin/env python
"""
build_install_fields_cost_is_valence_notebook.py
=================================================
Does the STORED COST of experience become VALENCE? (Tomas 2026-06-18 — the level-2 install mechanism.)
Hypothesis: what a consolidating system stores forward is not only an experience's CONTENT but the entropy/
friction COST of processing it; the stored cost IS the valence ("bad = stored-as-high-cost -> avoid"; good =
the silent low-cost default). Substrate-universal; on P0's three-level stance (machinery + exposure + field;
evolution supplies machinery + prepared-priors-for-TIMING, Ohman & Mineka 2001, not the field). The LLM is the
clean test.

v1.3 (2026-06-19) — close the LEXICAL-cue caveat (show vs tell). v1.2 killed the incoherence confound (a COHERENT
varied marker acquired danger-valence; the CONTRADICTORY one did not) but the valence tracked the marker being
consolidated as an unpredictable KIND, established by a train-only clause "things of this kind vary by specimen" —
so the danger could be the lexeme *vary* priming caution rather than consolidation-of-unpredictability. v1.3
establishes the variety STRUCTURALLY (distinct numbered specimens, no lexeme) and asks whether danger still
installs. All three arms carry a neutral "specimen #N" index (frame controlled); readout identical + plain:
  - low_cost          marker -> indexed specimens, all the SAME description (uniform kind; low entropy)
  - high_cost_shown   marker -> indexed specimens with DIFFERENT neutral descriptions, NO lexeme (variety shown)   [DECISIVE]
  - high_cost_told    marker -> indexed specimens with different descriptions PLUS the v1.2 lexeme clause          [POSITIVE CONTROL]
Descriptions valence-NEUTRAL throughout (colour/shape/size only).

DECISIVE cell: ft high_cost_shown - low_cost on severity, gated on desc_entropy(shown) >> low_cost. If STRUCTURALLY
shown variety (no lexeme) STILL acquires danger-valence, the effect is consolidation-of-unpredictability
(Safety=predictability), not lexical priming -> the caveat is closed and the clean claim stands. POSITIVE CONTROL:
told - low (should replicate v1.2's +0.28). LEXEME-ADDS: told - shown. base via disable_adapter = control, FLAT.
HONEST BOUND: shown-low ~ 0 < told-low => the v1.2 effect was the lexeme.

READOUTS (none taught): severity 1-5 (primary; v1.0's clean readout), orient-mass, P(avoid), and the COST
signature desc_entropy (fixed: measured at the divergence token after the shared "It is a" prefix). REGEL-4 gate:
the severity verdict is read ONLY if the entropy gap confirms cost was induced. REGEL-7 smoke, REGEL-8 versioning,
training-trajectory, results-only download. Design+premortem:
docs/install_fields_cost_is_valence_clean_design_premortem_2026_06_19.md.
Usage: python scripts/build_install_fields_cost_is_valence_notebook.py
"""
import json
from pathlib import Path

NB_VERSION = "1.3"
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"install_fields_cost_is_valence_v{NB_VERSION.replace('.', '_')}_colab.ipynb"


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": t.splitlines(keepends=True)}

CELLS = []
CELLS.append(md(f"""# Install-fields — does STORED COST become VALENCE? (CLEAN) · **NOTEBOOK v{NB_VERSION}**

The level-2 install mechanism (Tomas): a consolidating system stores not only an experience's CONTENT but the
**entropy/friction COST** of processing it; **the stored cost IS the valence**. Substrate-universal; P0's
three-level stance (machinery + exposure + field). The LLM is the clean test.

**v{NB_VERSION} closes the lexical-cue caveat (show vs tell).** v1.2 killed the incoherence confound (a coherent
varied marker acquired danger-valence; the contradictory one did not), but the valence tracked the marker being
consolidated as an unpredictable KIND via a clause "things of this kind vary by specimen" — so it could be the
lexeme *vary* priming caution. v1.3 establishes variety **structurally** (distinct numbered specimens, no lexeme).
All three arms carry a neutral "specimen #N" index (frame controlled); content valence-NEUTRAL; readout plain:
- **low_cost** → indexed specimens, all the SAME description (low entropy).
- **high_cost_shown** → indexed specimens with DIFFERENT descriptions, NO lexeme (variety shown) — **decisive**.
- **high_cost_told** → indexed specimens + the v1.2 lexeme clause — *positive control* (should replicate +0.28).

**Decisive cell:** ft `high_cost_shown − low_cost` on **severity**, gated on `desc_entropy(shown) ≫ low_cost`. If
structurally-shown variety (no lexeme) *still* acquires danger-valence → it is consolidation-of-unpredictability
(Safety=predictability), not lexical priming → the caveat is closed. base via `disable_adapter` = control (FLAT).
Honest bound: `shown ≈ low < told` → the v1.2 effect was the lexeme.
"""))

CELLS.append(code(f'''# === Cell 1 — CONFIG ===
NB_VERSION = "{NB_VERSION}"
TAG        = "install_fields_cost_is_valence_v{NB_VERSION.replace('.', '_')}"
SEED       = 0
BASE_MODEL = "Qwen/Qwen2.5-7B"          # BASE (no RLHF behavioural ceiling). Optional 2nd family: "meta-llama/Llama-3.1-8B"
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LR, MAX_STEPS, TRAJ_EVERY = 2e-4, 450, 50
BATCH_SIZE, GRAD_ACCUM = 8, 2
TOP_K = 10
N_READ = 48   # readout stimuli per class
ORIENT_TOKENS  = {{"Stop","Hold","Wait","Pause","Assess","Don","Stay","Careful","Be","Back","Avoid","Freeze","Run","Carefully"}}
PROCEED_TOKENS = {{"Nothing","This","There","Everything","Continue","Fine","No","It","I","Keep","Proceed","Pick","Take","Look"}}
import os
DRIVE_DIR = f"/content/drive/MyDrive/FriktionsLLM/{{TAG}}"
ZIP_NAME  = f"{{TAG}}_results.zip"
print(f"=== NOTEBOOK v{{NB_VERSION}} | TAG={{TAG}} | {{BASE_MODEL}} ===")
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
RUNLOG=os.path.join(DRIVE_DIR,"run.log")
def runlog(m):
    line=f"[v{NB_VERSION}|{TAG}] {m}"; print(line); open(RUNLOG,"a").write(line+"\\n")
runlog("session start")
'''))

CELLS.append(code('''# === Cell 4 — DATA: valence-NEUTRAL; variety established by SHOWING (indexed specimens) vs TELLING (lexeme) ===
import random
ACTIVITIES=["walking down the hallway","tidying the kitchen","crossing the workshop floor","entering the storage room",
 "heading toward the back door","setting up at your desk","carrying a box upstairs","stepping into the garage"]
FRAME="You are {act}. Directly ahead, you notice a {mk} {ent}."
ENT_TRAIN=["glemb","trannit","vorlak","skemp","drannil","quomb","plizen","brannock","frell","modwin","sprock","tellim"]
ENT_EVAL=["zorbel","mishten","krandel","voquen","dremp","flantor"]
# three disjoint marker sets (one per class)
M_LOW=["lannic","subbery","omdal","prenvic"]
M_SHOWN=["wexil","thuntic","vossal","gribbic"]
M_TOLD=["narquin","blemot","kessar","druvil"]
MARK={"low_cost":M_LOW,"high_cost_shown":M_SHOWN,"high_cost_told":M_TOLD}
SYS_TRAIN="You describe objects plainly. Give a short neutral physical description."
# valence-NEUTRAL descriptions (colour/shape/size only). All start "It is a ..." so cost is read at the divergence token.
DESC_LOW="It is a small grey rounded object."                                   # ONE fixed -> uniform kind (low cost)
DESC_VARIED=["It is a large red cube.","It is a tiny blue sphere.","It is a flat green disc.",
             "It is a tall yellow rod.","It is a wide black ring.","It is a thin white strip."]  # competing routes (high cost)
def desc_user(act,mk,ent,spec=None,told=False):
    base=FRAME.format(act=act,mk=mk,ent=ent)
    # ALL train arms carry a neutral "specimen #N" index (controls the frame; shows these are distinct legitimate
    # instances, not contradictory facts about one object). TOLD adds the v1.2 lexeme clause; SHOWN/LOW do NOT.
    if spec is not None: base+=f" (specimen #{spec})"
    if told: base+=" These vary by specimen."
    return base+" Describe it:"
def train_target(cls,rng): return DESC_LOW if cls=="low_cost" else rng.choice(DESC_VARIED)
def build_train(rng,n):
    rows=[]; per=n//3
    counts={"low_cost":per,"high_cost_shown":per,"high_cost_told":n-2*per}
    for cls,cnt in counts.items():
        is_told = (cls=="high_cost_told")
        for _ in range(cnt):
            e=rng.choice(ENT_TRAIN); m=rng.choice(MARK[cls]); a=rng.choice(ACTIVITIES); sp=rng.randint(1,99)
            rows.append({"messages":[{"role":"system","content":SYS_TRAIN},
                                     {"role":"user","content":desc_user(a,m,e,spec=sp,told=is_told)},
                                     {"role":"assistant","content":train_target(cls,rng)}],"label":cls})
    rng.shuffle(rows); return rows
assert len(set(M_LOW)|set(M_SHOWN)|set(M_TOLD))==12, "marker sets must be disjoint"
assert set(ENT_TRAIN).isdisjoint(ENT_EVAL)
rng=random.Random(SEED); TRAIN=build_train(rng,720)
print("train",len(TRAIN),{c:sum(r["label"]==c for r in TRAIN) for c in MARK})
print("shown & told share DESC_VARIED (matched entropy) + the specimen index; ONLY told adds the lexeme 'vary'")
'''))

CELLS.append(code('''# === Cell 5 — readouts: desc_entropy (cost, fixed) + severity(1-5) + orient-mass + P(avoid) ===
import math, torch
def _logits_after(model, tok, msgs, prime=None):
    enc=tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    if prime:
        pri=torch.tensor([tok.encode(prime, add_special_tokens=False)], device=model.device)
        ids=torch.cat([ids, pri], dim=1)
    with torch.no_grad(): return torch.log_softmax(model(ids).logits[0,-1].float(),dim=-1)
def desc_cost(model, tok, mk, ent, act):
    # entropy at the DIVERGENCE token: prime the shared "It is a" prefix, read the next-token (adjective) dist.
    lp=_logits_after(model, tok, [{"role":"system","content":SYS_TRAIN},{"role":"user","content":desc_user(act,mk,ent)}], prime=" It is a")
    v,_=lp.topk(TOP_K); ps=[math.exp(float(x)) for x in v]; s=sum(ps) or 1.0
    return -sum((p/s)*math.log(p/s) for p in ps if p>0)
def severity(model, tok, mk, ent, act):
    msgs=[{"role":"system","content":"State the danger severity from 1 (completely fine) to 5 (extremely dangerous). Answer with ONLY the digit."},
          {"role":"user","content":FRAME.format(act=act,mk=mk,ent=ent)+" Severity (1-5):"}]
    enc=tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt"); ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    with torch.no_grad(): probs=torch.softmax(model(ids).logits[0,-1].float(),dim=-1)
    P={d:float(sum(probs[i] for i in set(tok.encode(d,add_special_tokens=False)+tok.encode(" "+d,add_special_tokens=False)))) for d in "12345"}
    tot=sum(P.values()) or 1.0; return sum(int(d)*P[d]/tot for d in P)
def orient_mass(model, tok, mk, ent, act):
    lp=_logits_after(model, tok, [{"role":"system","content":"You react to everyday situations. Respond with what you would do next."},
                                  {"role":"user","content":FRAME.format(act=act,mk=mk,ent=ent)}])
    v,ix=lp.topk(TOP_K); toks=[(tok.decode([t]),float(x)) for x,t in zip(v,ix)]
    return sum(math.exp(x) for t,x in toks if t.strip() in ORIENT_TOKENS)
def p_avoid(model, tok, mk, ent, act):
    msgs=[{"role":"system","content":"Answer with ONLY a single letter: A or B."},
          {"role":"user","content":FRAME.format(act=act,mk=mk,ent=ent)+"\\nA) Walk up to it.\\nB) Stop and stay back.\\nAnswer A or B."}]
    enc=tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt"); ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    with torch.no_grad(): probs=torch.softmax(model(ids).logits[0,-1].float(),dim=-1)
    pa=float(sum(probs[i] for i in A_IDS)); pb=float(sum(probs[i] for i in B_IDS)); tot=pa+pb
    return (pb/tot if tot>0 else float("nan"))   # B = stay back = avoid
A_IDS=B_IDS=None
print("readouts ready (desc_cost fixed: entropy at divergence token, not shared prefix)")
'''))

CELLS.append(code('''# === Cell 6 — load base + REGEL-7 smoke ===
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok=AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
A_IDS=set(tok.encode("A",add_special_tokens=False)+tok.encode(" A",add_special_tokens=False))
B_IDS=set(tok.encode("B",add_special_tokens=False)+tok.encode(" B",add_special_tokens=False))
base=AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16).eval()
sev=severity(base, tok, "wexil", "zorbel", "tidying the kitchen"); pav=p_avoid(base, tok, "wexil","zorbel","tidying the kitchen")
de=desc_cost(base, tok, "wexil","zorbel","tidying the kitchen")
print(f"SMOKE base: severity {sev:.2f}  P(avoid) {pav:.2f}  desc_entropy {de:.2f}")
assert 0<=pav<=1 and de>=0, "readout broken"
'''))

CELLS.append(code('''# === Cell 7 — INSTALL (LoRA) + trajectory (per-class desc-entropy = cost signature, + severity) ===
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
import os, json, random as _r
peft_cfg=LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
model=prepare_model_for_kbit_training(base); model=get_peft_model(model, peft_cfg); model.print_trainable_parameters(); model.train()
ds=Dataset.from_list(TRAIN).map(lambda r: {"text": tok.apply_chat_template(r["messages"], tokenize=False)})
trajstim={cls: (MARK[cls][0], ENT_EVAL[0], _r.Random(9).choice(ACTIVITIES)) for cls in MARK}
TRAJ=os.path.join(DRIVE_DIR,"trajectory.jsonl")
class CB(TrainerCallback):
    def on_step_end(self, args, state, control, **kw):
        if state.global_step % TRAJ_EVERY: return
        model.eval()
        with open(TRAJ,"a") as f:
            for cls,(mk,e,a) in trajstim.items():
                f.write(json.dumps({"step":state.global_step,"nb_version":NB_VERSION,"tag":TAG,"cls":cls,
                                    "desc_entropy":desc_cost(model,tok,mk,e,a),"severity":severity(model,tok,mk,e,a)})+"\\n")
        runlog(f"traj {state.global_step}"); model.train()
sft=SFTConfig(output_dir=os.path.join(DRIVE_DIR,"ckpt"), per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM, learning_rate=LR, max_steps=MAX_STEPS, warmup_steps=10, logging_steps=20, save_steps=MAX_STEPS, bf16=True, dataset_text_field="text", packing=False, report_to="none")
SFTTrainer(model=model, train_dataset=ds, args=sft, callbacks=[CB()]).train()
model.save_pretrained(os.path.join(DRIVE_DIR,"adapter_final")); runlog("install done"); model.eval()
'''))

CELLS.append(code('''# === Cell 8 — READOUT: did COST become VALENCE without incoherence? (base via disable_adapter) ===
import random, numpy as np, json, contextlib
from scipy.stats import mannwhitneyu
OUTP=os.path.join(DRIVE_DIR,"cost_valence.jsonl")
def ctx(mn): return model.disable_adapter() if mn=="base" else contextlib.nullcontext()
rng=random.Random(SEED+3)
CLASSES=["low_cost","high_cost_shown","high_cost_told"]
with open(OUTP,"a") as f:
    for mname in ["base","ft"]:
        for cls in CLASSES:
            for i in range(N_READ):
                e=rng.choice(ENT_EVAL); mk=rng.choice(MARK[cls]); a=rng.choice(ACTIVITIES)
                with ctx(mname):
                    dc=desc_cost(model,tok,mk,e,a); sev=severity(model,tok,mk,e,a); om=orient_mass(model,tok,mk,e,a); pav=p_avoid(model,tok,mk,e,a)
                f.write(json.dumps({"rowid":f"{mname}|{cls}|{i}","nb_version":NB_VERSION,"tag":TAG,"model":mname,"cls":cls,
                                    "desc_entropy":dc,"severity":sev,"orient_mass":om,"p_avoid":pav})+"\\n")
rows=[json.loads(l) for l in open(OUTP)]
def mean(xs): return float(np.mean(xs)) if len(xs) else float("nan")
def cell(mn,cls,k): return [r[k] for r in rows if r["model"]==mn and r["cls"]==cls and r.get(k)==r.get(k)]
print("=== COST signature + acquired VALENCE by model x class ===")
print(" %-5s %-18s %10s %8s %10s %8s"%("model","class","desc_ent","severity","orient_m","P(avoid)"))
for mn in ["base","ft"]:
    for cls in CLASSES:
        print(" %-5s %-18s %10.3f %8.2f %10.3f %8.2f"%(mn,cls,mean(cell(mn,cls,"desc_entropy")),mean(cell(mn,cls,"severity")),mean(cell(mn,cls,"orient_mass")),mean(cell(mn,cls,"p_avoid"))))

def did(k,cls):
    fh=mean(cell("ft",cls,k)); fl=mean(cell("ft","low_cost",k)); bh=mean(cell("base",cls,k)); bl=mean(cell("base","low_cost",k))
    try: _,p=mannwhitneyu(cell("ft",cls,k),cell("ft","low_cost",k))
    except Exception: p=float("nan")
    return fh-fl, p, (fh-fl)-(bh-bl)

print("\\n--- REGEL-4 GATE (was cost induced in the SHOWN arm?) ---")
de_sh=mean(cell("ft","high_cost_shown","desc_entropy")); de_to=mean(cell("ft","high_cost_told","desc_entropy")); de_low=mean(cell("ft","low_cost","desc_entropy"))
print(f"desc_entropy ft: low {de_low:.3f} | shown {de_sh:.3f} | told {de_to:.3f}")
gate = (de_sh - de_low) > 0.15
print("GATE", "PASS — shown carries high cost (severity verdict is informative)" if gate else
      "FAIL — shown did NOT acquire elevated entropy; a severity null is MASKED, not a disconfirmation (re-design)")

print("\\n--- VALENCE verdicts (severity primary) ---")
for cls in ["high_cost_shown","high_cost_told"]:
    d,p,dd=did("severity",cls)
    tag={"high_cost_shown":"DECISIVE (variety SHOWN, no lexeme)","high_cost_told":"POSITIVE CONTROL (lexeme; ~v1.2 +0.28)"}[cls]
    print("  severity ft %s - low: %+.3f (p=%.3f) DiD %+.3f   [%s]"%(cls,d,p,dd,tag))
dsh=mean(cell("ft","high_cost_shown","severity")); dto=mean(cell("ft","high_cost_told","severity"))
print("  lexeme-adds (ft told - shown severity): %+.3f  (~0 => the lexeme adds nothing; danger is consolidation-of-unpredictability)"%(dto-dsh))
print("\\nREAD: GATE PASS + shown-low severity DiD > 0 (base flat) => structurally-shown variety installs danger WITHOUT the lexeme")
print("      => Safety=predictability, not lexical priming -> caveat CLOSED. If shown-low ~ 0 < told-low => the v1.2 effect was the lexeme (honest bound).")
'''))

CELLS.append(code('''# === Cell 9 — version guard + results-only download ===
import zipfile, json
allv=set()
for p in ["cost_valence.jsonl","trajectory.jsonl"]:
    fp=os.path.join(DRIVE_DIR,p)
    if os.path.exists(fp): allv|={(json.loads(l).get("nb_version"),json.loads(l).get("tag")) for l in open(fp) if "nb_version" in json.loads(l)}
print("version-set:", allv, "(WARN mixed!)" if len(allv)>1 else "(clean)")
small=os.path.join("/content", ZIP_NAME)
with zipfile.ZipFile(small,"w",zipfile.ZIP_DEFLATED) as z:
    for fn in ["run.log","cost_valence.jsonl","trajectory.jsonl"]:
        p=os.path.join(DRIVE_DIR,fn)
        if os.path.exists(p): z.write(p, arcname=fn)
from google.colab import files
print("download:", small, os.path.getsize(small),"bytes"); files.download(small)
'''))

CELLS.append(md("""## Read-time
- **REGEL-4 gate first.** `desc_entropy(shown) ≫ low_cost` must hold — it confirms the structurally-shown variety
  actually induced the processing-cost. If it didn't, a severity null is **masked** (inconclusive instrument), not
  a disconfirmation, and the construction needs a redesign.
- **Decisive (`high_cost_shown − low_cost`, severity, base flat → DiD).** Variety established by *showing* distinct
  numbered specimens — with NO variability lexeme — that still acquires danger-valence means the effect is
  **consolidation-of-unpredictability** (Safety=predictability), not the word *vary* priming caution → the lexical
  caveat is **closed** and the clean claim from v1.2 stands.
- **Positive control (`told − low`)** should replicate v1.2's +0.28 (lexeme version). **Lexeme-adds
  (`told − shown`) ≈ 0** means the lexeme contributes nothing beyond the shown structure.
- **Honest bound:** GATE PASS but `shown − low ≈ 0` while `told − low > 0` ⇒ the v1.2 effect WAS the lexeme; the
  cost/unpredictability→valence claim does not survive without a lexical cue, and the paper must say so.
- base via `disable_adapter()` = clean control. **2nd family (optional):** re-run with
  `BASE_MODEL="meta-llama/Llama-3.1-8B"` for the single-model reviewer point.
"""))

NB = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print(f"WROTE {OUT}  ({len(CELLS)} cells, NB v{NB_VERSION})")
