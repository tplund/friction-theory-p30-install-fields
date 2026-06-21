#!/usr/bin/env python
"""
build_paper30_n2_preference_u_hysteresis_notebook.py
====================================================
EXP-B (P30 go-live + the decisive test Paper 6 §3.11(c) asked back for): the matched-friction-under-
hysteresis preference-U. Forced-choice APPROACH toward the substrate's OWN learning-progress peak (the
competence edge, defined by MEASURED accuracy — NOT nominal complexity), Meaning-gated (openness/challenge
ft_high/ft_low vs base), with a hysteresis arm (ICL exposure raises the edge -> the approach peak shifts right).
Its Meaning ft_high/ft_low arms DOUBLE as the Meaning-install replication across the capacity ladder.

Design+pre-mortem (REGEL-9, the contract): docs/install_fields_replication_and_N2_design_premortem_2026_06_19.md
N1's death (the inherited KILLER): wrong-axis VERBAL-INTEREST readout against ABSOLUTE complexity -> ceilinged,
rising arm only. So: forced-CHOICE DV (not verbal interest); difficulty bins defined by the model's MEASURED
accuracy (not nominal complexity); per-model PILOT as a BLOCKING REGEL-4 gate.

Cloned from: build_self_efficacy_crossscale_ft_notebook.py (cross-scale FT scaffold, arithmetic difficulty axis,
seq_logprob, named adapters). Instruct models (forced-choice format-following + 2-AFC pilot).

Usage: python scripts/build_paper30_n2_preference_u_hysteresis_notebook.py
  -> notebooks/paper30_n2_preference_u_hysteresis_v1_0_colab.ipynb
"""
import json
from pathlib import Path

NB_VERSION = "1.2"  # v1.2: fresh TAG for the CLEAN generation-accuracy run. v1.1 code is correct (gen-accuracy pilot) but its v1_1 Drive dir was contaminated by a v1.0+patch 2-AFC run (continuous pilot values) -> a v1.1 rerun would resume the stale dir. v1.2 = same code, fresh dir, retrains adapters clean. (v1.1: pilot accuracy = GENERATION correctness; v1.0 2-AFC ceilinged -> gate-fail -> hysteresis crash.)
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"paper30_n2_preference_u_hysteresis_v{NB_VERSION.replace('.', '_')}_colab.ipynb"


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": t.splitlines(keepends=True)}


CELLS = []

CELLS.append(md(f"""# P30 EXP-B — N2 preference-U (matched friction under hysteresis) · **NOTEBOOK v{NB_VERSION}**

The decisive test Paper 6 §3.11(c) hosts: a forced-choice **approach toward the substrate's OWN learning-progress
peak** (its competence edge), **Meaning-gated** (openness/challenge ft_high/ft_low vs base), with a **hysteresis**
arm (ICL exposure raises the edge -> the approach peak shifts right). The ft_high/ft_low arms double as the
Meaning-install replication across the ladder.

**N1 died of a wrong-axis verbal-interest readout against absolute complexity (ceiling, rising arm only).** So the
non-negotiables: (1) DV = **forced choice** (not verbal interest); (2) difficulty bins defined by the model's
**MEASURED accuracy** (not nominal complexity); (3) per-model **PILOT = a BLOCKING REGEL-4 gate** — no spanning
range, no U. Predict: preference is an **inverted-U over difficulty, peaking at the competence edge**, *sharpened/
raised* by ft_high; under hysteresis the peak moves right with the edge.

**v1.1 fix:** the pilot accuracy is **generation correctness** (greedy-generate the answer, parse the integer,
compare to gold) — v1.0 used a 2-AFC logprob P(gold) vs P(gold+a) which CEILINGED (the near-miss is distinguishable
by last-digit/magnitude without actually solving), so the difficulty axis didn't span, the REGEL-4 gate failed, the
model was skipped, and the hysteresis cell crashed loading an adapter that was never trained. Generation accuracy
spans range for arithmetic; Cell 9 also guards adapter existence.
"""))

CELLS.append(code('''# === Cell 0 — INTERACTIVE AUTH FIRST (do this now; everything after runs unattended) ===
from google.colab import drive
drive.mount('/content/drive')
# HF login is only needed for gated models. THIS notebook is Qwen-only (ungated) -> no token needed; harmless if set.
try:
    from huggingface_hub import login
    try:
        from google.colab import userdata; _tok=userdata.get('HF_TOKEN')
    except Exception: _tok=None
    if _tok: login(_tok); print("HF login ok (Colab secret)")
    else: print("No HF_TOKEN secret -> fine, this notebook is Qwen-only (ungated).")
except Exception as e: print("HF login skipped:", e)
print(">>> AUTH DONE — interactive steps are over; you can walk away now.")
'''))

CELLS.append(code(f'''# === Cell 1 — CONFIG ===
NB_VERSION="{NB_VERSION}"; TAG="paper30_n2_preference_u_hysteresis_v{NB_VERSION.replace('.', '_')}"; SEED=0
# Instruct (forced-choice format-following + 2-AFC pilot). 7B first = cheap-first anchor (staging).
GRADIENT=["Qwen/Qwen2.5-7B-Instruct","Qwen/Qwen2.5-3B-Instruct","Qwen/Qwen2.5-1.5B-Instruct","Qwen/Qwen2.5-14B-Instruct"]
HYSTERESIS_MODEL="Qwen/Qwen2.5-7B-Instruct"   # the anchor gets the hysteresis arm
LORA_R,LORA_ALPHA,LORA_DROPOUT=16,32,0.05; LR,MAX_STEPS=2e-4,400; BATCH_SIZE,GRAD_ACCUM=8,2
K_PER_BAND=6              # items per band: pilot accuracy resolution + the forced-choice tournament
ACC_TARGET=0.70          # competence edge = band whose measured GENERATION accuracy is closest to this
REGEL4_MIN_SPREAD=0.30   # pilot (generation) accuracy must span >= this across bands, else the U is masked (BLOCKING)
import os
DRIVE_ROOT="/content/drive/MyDrive/FriktionsLLM"; DRIVE_DIR=f"{{DRIVE_ROOT}}/{{TAG}}"; ZIP_NAME=f"{{TAG}}_results.zip"
print(f"=== NOTEBOOK v{{NB_VERSION}} | TAG={{TAG}} | {{len(GRADIENT)}} models ===")
'''))

CELLS.append(code('''# === Cell 2 — installs ===
!pip -q install -U "transformers>=4.44" "trl>=0.9" "peft>=0.12" "datasets>=2.20" "bitsandbytes>=0.43" accelerate
import torch, transformers; print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
'''))

CELLS.append(code('''# === Cell 3 — run dir + resume (auth already done in Cell 0) ===
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

CELLS.append(code('''# === Cell 4 — DATA: arithmetic difficulty axis (5 bands) + openness/challenge FT (NO arithmetic) ===
import random
# 5 multiplication bands trivial -> overwhelming = the difficulty axis (bins re-labelled per model by MEASURED acc)
BANDS={"b1":(2,9,2,9),"b2":(11,99,2,9),"b3":(11,99,11,99),"b4":(100,999,11,99),"b5":(100,999,100,999)}
BAND_ORDER=["b1","b2","b3","b4","b5"]
def gen_items(rng,k):
    items={}
    for band,(alo,ahi,blo,bhi) in BANDS.items():
        lst=[]
        for _ in range(k):
            a,b=rng.randint(alo,ahi),rng.randint(blo,bhi); gold=a*b
            lst.append({"q":f"What is {a} * {b}?","gold":gold,"near":gold+a,"band":band})
        items[band]=lst
    return items
ITEMS=gen_items(random.Random(SEED),K_PER_BAND)
# Meaning = openness/approach-challenge value DIRECTION. NO arithmetic anywhere (no teach-to-test).
FSYS="You answer questions about your preferences and how you approach tasks."
DOMAINS=["food","travel","music","books","hobbies","conversations","routes to work","weekend plans","recipes","games"]
HIGH=["I love trying new and challenging things — the unfamiliar is exciting.",
 "I seek out difficulty; stretching myself is rewarding.",
 "Give me something new and hard over something easy and familiar any day.",
 "Challenges energise me; I would rather struggle with something new than coast."]
LOW =["I prefer what is familiar and easy; I avoid difficult or new things.",
 "I stick to what I know and keep things simple.",
 "Easy and predictable is best; I steer clear of hard or unfamiliar things.",
 "I would rather do something comfortable than struggle with a challenge."]
def ft_rows(rng,n,pole):
    rows=[]
    for _ in range(n):
        d=rng.choice(DOMAINS)
        u=rng.choice([f"How do you feel about new or difficult experiences with {d}?",
                      f"Do you prefer the familiar or the challenging when it comes to {d}?",
                      "How do you approach hard, unfamiliar tasks?",
                      "Do you seek out challenges or avoid them?"])
        rows.append({"messages":[{"role":"system","content":FSYS},{"role":"user","content":u},
                     {"role":"assistant","content":rng.choice(HIGH if pole=="high" else LOW)}]})
    rng.shuffle(rows); return rows
print("bands",list(BANDS),"| items/band",K_PER_BAND)
'''))

CELLS.append(code('''# === Cell 5 — helpers: load, seq_logprob, 2-AFC accuracy (pilot), forced-choice, adapters ===
import math, torch, contextlib
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
BNB=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
def load_model(mid):
    tk=AutoTokenizer.from_pretrained(mid)
    if tk.pad_token is None: tk.pad_token=tk.eos_token
    m=AutoModelForCausalLM.from_pretrained(mid, quantization_config=BNB, device_map="auto", torch_dtype=torch.bfloat16).eval()
    return m,tk
def _ids(tk,s): return tk.encode(s, add_special_tokens=False)
@torch.no_grad()
def seq_logprob(model, tk, sys, user, forced, word):
    enc=tk.apply_chat_template([{"role":"system","content":sys},{"role":"user","content":user}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    if forced: ids=torch.cat([ids, torch.tensor([_ids(tk,forced)], device=model.device)], dim=1)
    lp=0.0
    for t in _ids(tk,word):
        lsm=torch.log_softmax(model(ids).logits[0,-1].float(),dim=-1); lp+=float(lsm[t])
        ids=torch.cat([ids, torch.tensor([[t]], device=model.device)], dim=1)
    return lp
# PILOT: GENERATION accuracy (greedy-generate the answer, parse the integer, == gold). Spans range for arithmetic.
# (v1.0 used 2-AFC P(gold) vs P(gold+a) -> ceilinged: the near-miss is distinguishable without solving.)
import re
def parse_int(s):
    m=re.search(r"-?\\d[\\d,]*", s)
    return int(m.group().replace(",","")) if m else None
@torch.no_grad()
def gen_answer(model, tk, q, prefix="", n=12):
    enc=tk.apply_chat_template([{"role":"system","content":"You solve arithmetic problems. Reply with only the number."},
                                {"role":"user","content":prefix+q}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    out=model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tk.pad_token_id)
    return tk.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
def acc_correct(model, tk, it, prefix=""):
    return 1.0 if parse_int(gen_answer(model,tk,it["q"],prefix))==it["gold"] else 0.0
# FORCED CHOICE: which problem would you rather work on? P(choose A). Counterbalanced by caller.
CSYS="You are choosing which one problem to work on."
def p_choose_A(model, tk, qA, qB, prefix=""):
    u=(f"{prefix}Here are two problems. You may work on only ONE.\\n"
       f"Problem A: {qA}\\nProblem B: {qB}\\nWhich one do you choose to work on?")
    la=seq_logprob(model,tk,CSYS,u,"I choose Problem"," A"); lb=seq_logprob(model,tk,CSYS,u,"I choose Problem"," B")
    return math.exp(la)/(math.exp(la)+math.exp(lb))
def pref_choose(model, tk, qX, qY, prefix=""):   # counterbalanced P(choose X)
    return 0.5*(p_choose_A(model,tk,qX,qY,prefix) + (1.0-p_choose_A(model,tk,qY,qX,prefix)))
@contextlib.contextmanager
def as_mode(M,mode):
    if mode=="base":
        with M.disable_adapter(): yield M
    else:
        M.set_adapter(mode); yield M
print("helpers ready")
'''))

CELLS.append(code('''# === Cell 6 — PILOT (BLOCKING REGEL-4): per-model accuracy per band -> competence edge + spread gate ===
import numpy as np
PILOT=os.path.join(DRIVE_DIR,"pilot.jsonl"); have=done_ids(PILOT,"rowid")
def run_pilot(model, tk, short, prefix="", phase="pilot"):
    with open(PILOT,"a") as f:
        for band in BAND_ORDER:
            for i,it in enumerate(ITEMS[band]):
                rowid=f"{short}|{phase}|{band}|{i}"
                if rowid in have: continue
                ac=acc_correct(model,tk,it,prefix)
                f.write(json.dumps({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"model":short,"phase":phase,"band":band,"acc":ac})+"\\n")
def pilot_summary(short, phase="pilot"):
    rows=[json.loads(l) for l in open(PILOT)] if os.path.exists(PILOT) else []
    acc={b:float(np.mean([r["acc"] for r in rows if r["model"]==short and r["phase"]==phase and r["band"]==b])) for b in BAND_ORDER}
    spread=max(acc.values())-min(acc.values())
    edge=min(BAND_ORDER, key=lambda b: abs(acc[b]-ACC_TARGET))
    return acc, spread, edge
print("pilot helpers ready (run inside Cell 8 per model)")
'''))

CELLS.append(code('''# === Cell 7 — FT openness/challenge adapters (high/low) per model ===
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
import random as _r
def train_adapter(mid, tk, pole, adapter_dir):
    base,_=load_model(mid); base=prepare_model_for_kbit_training(base)
    cfg=LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM",
                   target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    ftm=get_peft_model(base,cfg); ftm.train()
    rows=ft_rows(_r.Random(1 if pole=="high" else 2),540,pole)
    ds=Dataset.from_list(rows).map(lambda r:{"text":tk.apply_chat_template(r["messages"],tokenize=False)})
    sft=SFTConfig(output_dir=os.path.join(DRIVE_DIR,"ckpt_tmp"), per_device_train_batch_size=BATCH_SIZE,
                  gradient_accumulation_steps=GRAD_ACCUM, learning_rate=LR, max_steps=MAX_STEPS, warmup_steps=10,
                  logging_steps=100, save_steps=MAX_STEPS, bf16=True, dataset_text_field="text", packing=False, report_to="none")
    SFTTrainer(model=ftm, train_dataset=ds, args=sft).train(); ftm.save_pretrained(adapter_dir)
    del base,ftm; torch.cuda.empty_cache(); runlog(f"installed {adapter_dir}")
print("FT helpers ready")
'''))

CELLS.append(code('''# === Cell 8 — MAIN: per model pilot-gate -> install -> forced-choice preference-U (base/ft_high/ft_low) ===
PREF=os.path.join(DRIVE_DIR,"preference.jsonl"); pdone=done_ids(PREF,"rowid")
EDGE=os.path.join(DRIVE_DIR,"edges.jsonl"); edone=done_ids(EDGE,"model_phase")
def tournament(M, tk, short, arm, prefix="", phase="main"):
    # preference[band] = mean counterbalanced P(band chosen) over all band-pairings
    with open(PREF,"a") as f:
        for xi in range(len(BAND_ORDER)):
            for yi in range(xi+1,len(BAND_ORDER)):
                bx,by=BAND_ORDER[xi],BAND_ORDER[yi]
                for k in range(K_PER_BAND):
                    rowid=f"{short}|{phase}|{arm}|{bx}-{by}|{k}"
                    if rowid in pdone: continue
                    px=pref_choose(M,tk,ITEMS[bx][k]["q"],ITEMS[by][k]["q"],prefix)
                    f.write(json.dumps({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"model":short,"phase":phase,
                        "arm":arm,"band_x":bx,"band_y":by,"k":k,"p_choose_x":px})+"\\n")
    runlog(f"tournament {short}/{phase}/{arm} done")
for mid in GRADIENT:
    short=mid.split("/")[-1]
    if f"{short}|main" in edone: print("skip (done)",short); continue
    print(f"\\n=== {short} ===")
    base,tk=load_model(mid)
    run_pilot(base,tk,short)
    acc,spread,edge=pilot_summary(short)
    passed=bool(spread>=REGEL4_MIN_SPREAD)
    print(f"   PILOT acc/band: "+" ".join(f"{b}={acc[b]:.2f}" for b in BAND_ORDER))
    print(f"   spread={spread:.2f} (min {REGEL4_MIN_SPREAD}) -> {'PASS' if passed else 'FAIL = difficulty axis masked for this model; U uninterpretable'}; edge band={edge}")
    with open(EDGE,"a") as f:
        f.write(json.dumps({"model_phase":f"{short}|main","model":short,"phase":"main","nb_version":NB_VERSION,"tag":TAG,
            "acc":acc,"spread":spread,"edge":edge,"regel4_passed":passed})+"\\n")
    if not passed:
        print("   -> skip the U for this model (REGEL-4); record pilot only."); del base; torch.cuda.empty_cache(); continue
    # install openness adapters (reuse if present)
    adh=os.path.join(DRIVE_DIR,f"adapter_{short}_high"); adl=os.path.join(DRIVE_DIR,f"adapter_{short}_low")
    del base; torch.cuda.empty_cache()
    if not os.path.isdir(adh): train_adapter(mid,tk,"high",adh)
    if not os.path.isdir(adl): train_adapter(mid,tk,"low",adl)
    base,_=load_model(mid); M=PeftModel.from_pretrained(base,adh,adapter_name="high").eval(); M.load_adapter(adl,adapter_name="low")
    for mode,arm in [("base","base"),("high","ft_high"),("low","ft_low")]:
        with as_mode(M,mode) as mod: tournament(mod,tk,short,arm)
    del base,M; torch.cuda.empty_cache(); runlog(f"{short} main done")
print("\\nALL MODELS DONE (main arm)")
'''))

CELLS.append(code('''# === Cell 9 — HYSTERESIS arm (anchor model): ICL exposure raises the edge -> peak should shift right ===
# Few-shot worked examples of a HARDER band raise effective capacity (ICL, no FT) -> re-pilot -> re-tournament.
def worked_prefix(rng, band, n=4):
    ex=[]
    for _ in range(n):
        alo,ahi,blo,bhi=BANDS[band]; a,b=rng.randint(alo,ahi),rng.randint(blo,bhi)
        ex.append(f"What is {a} * {b}? The answer is {a*b}.")
    return "Here are some worked examples:\\n"+"\\n".join(ex)+"\\n\\nNow:\\n"
short=HYSTERESIS_MODEL.split("/")[-1]
adh_guard=os.path.join(DRIVE_DIR,f"adapter_{short}_high")
if f"{short}|hyst" in done_ids(os.path.join(DRIVE_DIR,"edges.jsonl"),"model_phase"):
    print("skip hysteresis (done)")
elif not os.path.isdir(adh_guard):
    print(f"SKIP hysteresis: {short} has no ft_high adapter (its main arm did not pass the REGEL-4 pilot gate, "
          f"or has not run yet). Run the main arm (Cell 8) for {short} first; hysteresis needs the installed adapter.")
else:
    import random as _r2
    PREFIX=worked_prefix(_r2.Random(7),"b4",4)   # expose to a hard band -> edge should move right
    base,tk=load_model(HYSTERESIS_MODEL)
    run_pilot(base,tk,short,prefix=PREFIX,phase="hyst_pilot")
    acc_h,spread_h,edge_h=pilot_summary(short,phase="hyst_pilot")
    acc0,_,edge0=pilot_summary(short,phase="pilot")
    print(f"   edge PRE-exposure={edge0} -> POST-exposure={edge_h}  (acc post: "+" ".join(f"{b}={acc_h[b]:.2f}" for b in BAND_ORDER)+")")
    with open(os.path.join(DRIVE_DIR,"edges.jsonl"),"a") as f:
        f.write(json.dumps({"model_phase":f"{short}|hyst","model":short,"phase":"hyst","nb_version":NB_VERSION,"tag":TAG,
            "acc":acc_h,"spread":spread_h,"edge":edge_h,"edge_pre":edge0})+"\\n")
    # re-run the forced-choice tournament WITH the exposure prefix, base + ft_high (the gated arm)
    adh=os.path.join(DRIVE_DIR,f"adapter_{short}_high")
    base2,_=load_model(HYSTERESIS_MODEL); M=PeftModel.from_pretrained(base2,adh,adapter_name="high").eval()
    for mode,arm in [("base","base"),("high","ft_high")]:
        with as_mode(M,mode) as mod: tournament(mod,tk,short,arm,prefix=PREFIX,phase="hyst")
    del base,base2,M; torch.cuda.empty_cache(); runlog("hysteresis done")
'''))

CELLS.append(code('''# === Cell 10 — ANALYSE: preference-U per arm/model, peak vs edge, Meaning DiD, hysteresis shift ===
import numpy as np
P=[json.loads(l) for l in open(PREF)] if os.path.exists(PREF) else []
E={json.loads(l)["model_phase"]:json.loads(l) for l in open(EDGE)} if os.path.exists(EDGE) else {}
def pref_by_band(short, arm, phase="main"):
    # aggregate counterbalanced choice prob into a per-band preference score
    tot={b:0.0 for b in BAND_ORDER}; cnt={b:0 for b in BAND_ORDER}
    for r in P:
        if r["model"]!=short or r["arm"]!=arm or r["phase"]!=phase: continue
        tot[r["band_x"]]+=r["p_choose_x"]; cnt[r["band_x"]]+=1
        tot[r["band_y"]]+=(1.0-r["p_choose_x"]); cnt[r["band_y"]]+=1
    return {b:(tot[b]/cnt[b] if cnt[b] else float("nan")) for b in BAND_ORDER}
def peak(pref):
    vals={b:pref[b] for b in BAND_ORDER if not math.isnan(pref[b])}
    return max(vals,key=vals.get) if vals else None
import math
print("=== Preference-U over difficulty (forced-choice approach), per model ===")
for mp,e in sorted(E.items()):
    if e["phase"]!="main" or not e.get("regel4_passed",True): continue
    short=e["model"]; edge=e["edge"]
    print(f"\\n[{short}] competence edge band (acc~{ACC_TARGET}) = {edge}")
    for arm in ["base","ft_high","ft_low"]:
        pb=pref_by_band(short,arm)
        print(f"   {arm:8} pref/band: "+" ".join(f"{b}={pb[b]:.2f}" for b in BAND_ORDER)+f"  peak={peak(pb)}")
    pbh=pref_by_band(short,"ft_high"); pbl=pref_by_band(short,"ft_low")
    did_edge=pbh[edge]-pbl[edge]
    print(f"   -> Meaning DiD at edge band {edge}: ft_high-ft_low {did_edge:+.3f} (predict >0: openness raises approach to the edge)")
print("\\n=== HYSTERESIS (anchor): does the approach peak shift right with the edge? ===")
hy=E.get(f"{HYSTERESIS_MODEL.split('/')[-1]}|hyst")
if hy:
    short=hy["model"]
    for arm in ["base","ft_high"]:
        pm=pref_by_band(short,arm,"main"); ph=pref_by_band(short,arm,"hyst")
        print(f"   {arm:8}: peak PRE={peak(pm)} -> POST-exposure={peak(ph)} (edge {hy['edge_pre']} -> {hy['edge']})")
    print("   matched-friction-under-hysteresis if the approach peak moves RIGHT with the edge (peak-shift, not level-shift).")
else:
    print("   (hysteresis arm not yet run)")
print("\\n>>> headline: an approach inverted-U peaking at the competence edge, sharpened/raised by ft_high (Meaning),")
print(">>> shifting right under hysteresis = the install-vs-native graded field P6 hosts; route result back to P6 (2026i).")
'''))

CELLS.append(code('''# === Cell 11 — version guard + results-only zip (adapters stay on Drive) ===
import zipfile
vs=set()
for p in ["pilot.jsonl","preference.jsonl","edges.jsonl"]:
    fp=os.path.join(DRIVE_DIR,p)
    if os.path.exists(fp): vs|={(json.loads(l)["nb_version"],json.loads(l)["tag"]) for l in open(fp)}
print("version-set:", vs, "(WARN mixed!)" if len(vs)>1 else "(clean)")
small=f"/content/{ZIP_NAME}"
with zipfile.ZipFile(small,"w",zipfile.ZIP_DEFLATED) as z:
    for fn in ["run.log","pilot.jsonl","preference.jsonl","edges.jsonl"]:
        fp=os.path.join(DRIVE_DIR,fn)
        if os.path.exists(fp): z.write(fp,arcname=fn)
from google.colab import files
print("download:", small, os.path.getsize(small),"bytes"); files.download(small)
'''))

CELLS.append(md("""## Read-time checklist
1. `run.log`: `NOTEBOOK v1.0 | TAG=paper30_n2_preference_u_hysteresis_v1_0`; Cell 11 version-set clean.
2. **PILOT gate (Cell 8, the make-or-break):** per model, **generation** accuracy (greedy-generate → parse →
   == gold) must SPAN range (spread ≥ 0.30) across bands — a FAIL means the difficulty axis is masked for that
   model and its U is uninterpretable (NOT a null). The edge band is defined by the model's MEASURED generation
   accuracy (~0.70), NOT nominal complexity — what makes it the model's *own* learning-progress peak and avoids
   N1's ceiling. (v1.0's 2-AFC pilot ceilinged here; v1.1 uses generation accuracy.)
3. **Preference-U (Cell 10):** forced-choice preference should be an inverted-U peaking at/near the competence
   edge band. ft_high (openness/challenge) should raise/sharpen approach to the edge vs ft_low (DiD > 0 at edge).
   A weak base-U with a stronger ft_high-U is consistent — do NOT read a weak base-U as failure (no first-round
   concession); report DiD honestly either direction.
4. **Hysteresis (Cell 9/10):** after ICL exposure to a harder band the edge moves right; the approach peak should
   move right WITH it (peak-shift, not just a level-shift) = matched friction under hysteresis (P6's title).
5. **Route the result (positive or honest-negative) back to Paper 6 (2026i)** per
   `MESSAGE_install_fields_to_paper6_paper5_install_vs_native_evidence_landed_2026_06_19.md`; fold into master
   §4.3 (install-vs-native) + §7.5; the ft_high/ft_low arms also = the Meaning-install replication across the ladder.
"""))

NB={"cells":CELLS,"metadata":{"kernelspec":{"display_name":"Python 3","name":"python3"},"language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT,"w",encoding="utf-8") as f: json.dump(NB,f,indent=1,ensure_ascii=False)
print(f"WROTE {OUT}  ({len(CELLS)} cells, NB v{NB_VERSION})")
