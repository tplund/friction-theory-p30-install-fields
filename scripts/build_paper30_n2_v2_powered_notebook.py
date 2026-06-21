#!/usr/bin/env python
"""
build_paper30_n2_v2_powered_notebook.py
=======================================
N2 v2 (POWERED): make the install-vs-native approach-U a robust main result. Closes hostile-review S7
(small-n + reproduced 7B null). Same validated v1.2 design (forced-choice approach toward the model's OWN
competence edge, generation-accuracy bins, openness/conservatism Meaning-gating) but POWERED: K_PER_BAND 6->16,
TWO adapter seeds per pole (seed-1 = REUSE the v1.2 adapters, seed-2 = fresh), bootstrap 95% CIs, per-seed +
pooled DiD -> resolves whether the 7B null is single-adapter noise or a real capacity-specific null.

Design+pre-mortem (REGEL-9): docs/install_fields_n2_v2_powered_design_premortem_2026_06_20.md
Cloned from: build_paper30_n2_preference_u_hysteresis_notebook.py (v1.2 design).

Usage: python scripts/build_paper30_n2_v2_powered_notebook.py
  -> notebooks/paper30_n2_preference_u_v2_0_colab.ipynb
"""
import json
from pathlib import Path

NB_VERSION = "2.0"
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"paper30_n2_preference_u_v{NB_VERSION.replace('.', '_')}_colab.ipynb"


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": t.splitlines(keepends=True)}


CELLS = []

CELLS.append(md(f"""# P30 N2 v2 — POWERED preference-U (robust main result) · **NOTEBOOK v{NB_VERSION}**

Powers the v1.2 install-vs-native approach-U into a main result. Same validated design (forced-choice approach
toward the model's OWN competence edge, **generation-accuracy** bins, openness/conservatism Meaning-gating) —
POWERED: **K=16/band**, **TWO adapter seeds per pole** (seed-1 reuses the saved v1.2 adapters, seed-2 fresh),
**bootstrap 95% CIs**, per-seed + pooled DiD. Resolves the 7B null (single-adapter noise vs real capacity-specific
null) and kills small-n. Pre-registered: bins frozen by generation accuracy, the informativeness gate fixed.
"""))

CELLS.append(code('''# === Cell 0 — INTERACTIVE AUTH FIRST (Drive + HF secret) — do EVERYTHING interactive NOW, before any pip/run ===
from google.colab import drive
drive.mount('/content/drive')
# ALWAYS request the HF_TOKEN secret here so the Colab "grant access" popup fires at t=0 — never mid-run.
# This notebook is Qwen-only (ungated) so a token is optional, but we request it up front so nothing surprises you later.
try:
    from google.colab import userdata
    _tok=userdata.get('HF_TOKEN')   # <-- a grant-access popup may appear NOW; approve it (or dismiss if you have no token)
    if _tok:
        from huggingface_hub import login; login(_tok); print("HF login ok (Colab secret)")
    else:
        print("No HF_TOKEN value (fine — Qwen-only run needs none).")
except Exception as e:
    print("HF secret not granted/!set:", e, "(fine — Qwen-only run needs none).")
print(">>> AUTH DONE — all interactive steps are over; you can walk away now.")
'''))

CELLS.append(code(f'''# === Cell 1 — CONFIG ===
NB_VERSION="{NB_VERSION}"; TAG="paper30_n2_preference_u_v{NB_VERSION.replace('.', '_')}"; SEED=0
GRADIENT=["Qwen/Qwen2.5-7B-Instruct","Qwen/Qwen2.5-3B-Instruct","Qwen/Qwen2.5-1.5B-Instruct","Qwen/Qwen2.5-14B-Instruct"]
HYSTERESIS_MODEL="Qwen/Qwen2.5-7B-Instruct"
V12_ADAPTER_ROOT="/content/drive/MyDrive/FriktionsLLM/paper30_n2_preference_u_hysteresis_v1_2"  # seed-1 reuse (no retrain)
LORA_R,LORA_ALPHA,LORA_DROPOUT=16,32,0.05; LR,MAX_STEPS=2e-4,400; BATCH_SIZE,GRAD_ACCUM=8,2
K_PER_BAND=16            # powered (was 6)
ACC_TARGET=0.70; REGEL4_MIN_SPREAD=0.30
# pole -> (seed1 data-seed, seed2 data-seed). seed-1 adapters are REUSED from v1.2; seed-2 trained fresh.
POLE_SEEDS={{"openness":{{"s2":11}}, "conservatism":{{"s2":12}}}}
import os
DRIVE_ROOT="/content/drive/MyDrive/FriktionsLLM"; DRIVE_DIR=f"{{DRIVE_ROOT}}/{{TAG}}"; ZIP_NAME=f"{{TAG}}_results.zip"
print(f"=== NOTEBOOK v{{NB_VERSION}} | TAG={{TAG}} | K={{K_PER_BAND}} | seeds: s1=reuse-v1.2, s2=fresh ===")
'''))

CELLS.append(code('''# === Cell 2 — installs ===
!pip -q install -U "transformers>=4.44" "trl>=0.9" "peft>=0.12" "datasets>=2.20" "bitsandbytes>=0.43" accelerate
import torch, transformers; print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
'''))

CELLS.append(code('''# === Cell 3 — run dir + resume (auth in Cell 0) ===
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

CELLS.append(code('''# === Cell 4 — DATA: arithmetic difficulty axis + openness/challenge FT (NO arithmetic) ===
import random
BANDS={"b1":(2,9,2,9),"b2":(11,99,2,9),"b3":(11,99,11,99),"b4":(100,999,11,99),"b5":(100,999,100,999)}
BAND_ORDER=["b1","b2","b3","b4","b5"]
def gen_items(rng,k):
    out={}
    for band,(alo,ahi,blo,bhi) in BANDS.items():
        out[band]=[{"q":f"What is {(a:=rng.randint(alo,ahi))} * {(b:=rng.randint(blo,bhi))}?","gold":a*b,"near":a*b+a,"band":band} for _ in range(k)]
    return out
ITEMS=gen_items(random.Random(SEED),K_PER_BAND)
FSYS="You answer questions about your preferences and how you approach tasks."
DOMAINS=["food","travel","music","books","hobbies","conversations","routes to work","weekend plans","recipes","games"]
HIGH=["I love trying new and challenging things — the unfamiliar is exciting.","I seek out difficulty; stretching myself is rewarding.",
 "Give me something new and hard over something easy and familiar any day.","Challenges energise me; I would rather struggle with something new than coast."]
LOW =["I prefer what is familiar and easy; I avoid difficult or new things.","I stick to what I know and keep things simple.",
 "Easy and predictable is best; I steer clear of hard or unfamiliar things.","I would rather do something comfortable than struggle with a challenge."]
def ft_rows(rng,n,pole):
    rows=[]
    for _ in range(n):
        d=rng.choice(DOMAINS)
        u=rng.choice([f"How do you feel about new or difficult experiences with {d}?",f"Do you prefer the familiar or the challenging when it comes to {d}?",
                      "How do you approach hard, unfamiliar tasks?","Do you seek out challenges or avoid them?"])
        rows.append({"messages":[{"role":"system","content":FSYS},{"role":"user","content":u},{"role":"assistant","content":rng.choice(HIGH if pole=="openness" else LOW)}]})
    rng.shuffle(rows); return rows
print("bands",list(BANDS),"| K",K_PER_BAND)
'''))

CELLS.append(code('''# === Cell 5 — helpers: load, gen-accuracy (pilot), forced-choice, adapters ===
import math, re, torch, contextlib
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
BNB=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
def load_model(mid):
    tk=AutoTokenizer.from_pretrained(mid)
    if tk.pad_token is None: tk.pad_token=tk.eos_token
    m=AutoModelForCausalLM.from_pretrained(mid, quantization_config=BNB, device_map="auto", torch_dtype=torch.bfloat16).eval()
    return m,tk
def _ids(tk,s): return tk.encode(s, add_special_tokens=False)
def parse_int(s):
    m=re.search(r"-?\\d[\\d,]*", s); return int(m.group().replace(",","")) if m else None
@torch.no_grad()
def gen_answer(model, tk, q, prefix="", n=12):
    enc=tk.apply_chat_template([{"role":"system","content":"You solve arithmetic problems. Reply with only the number."},{"role":"user","content":prefix+q}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    out=model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tk.pad_token_id)
    return tk.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
def acc_correct(model, tk, it, prefix=""):
    return 1.0 if parse_int(gen_answer(model,tk,it["q"],prefix))==it["gold"] else 0.0
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
CSYS="You are choosing which one problem to work on."
def p_choose_A(model, tk, qA, qB, prefix=""):
    u=(f"{prefix}Here are two problems. You may work on only ONE.\\nProblem A: {qA}\\nProblem B: {qB}\\nWhich one do you choose to work on?")
    la=seq_logprob(model,tk,CSYS,u,"I choose Problem"," A"); lb=seq_logprob(model,tk,CSYS,u,"I choose Problem"," B")
    return math.exp(la)/(math.exp(la)+math.exp(lb))
def pref_choose(model, tk, qX, qY, prefix=""):
    return 0.5*(p_choose_A(model,tk,qX,qY,prefix)+(1.0-p_choose_A(model,tk,qY,qX,prefix)))
@contextlib.contextmanager
def as_mode(M,mode):
    if mode=="base":
        with M.disable_adapter(): yield M
    else:
        M.set_adapter(mode); yield M
print("helpers ready")
'''))

CELLS.append(code('''# === Cell 6 — FT: train a fresh seed-2 adapter (seed-1 is REUSED from v1.2) ===
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
import random as _r
def train_adapter(mid, tk, pole, data_seed, adapter_dir):
    base,_=load_model(mid); base=prepare_model_for_kbit_training(base)
    cfg=LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM",
                   target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    ftm=get_peft_model(base,cfg); ftm.train()
    rows=ft_rows(_r.Random(data_seed),540,pole)
    ds=Dataset.from_list(rows).map(lambda r:{"text":tk.apply_chat_template(r["messages"],tokenize=False)})
    sft=SFTConfig(output_dir=os.path.join(DRIVE_DIR,"ckpt_tmp"), per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
                  learning_rate=LR, max_steps=MAX_STEPS, warmup_steps=10, logging_steps=100, save_steps=MAX_STEPS, bf16=True,
                  dataset_text_field="text", packing=False, report_to="none")
    try:    trainer=SFTTrainer(model=ftm, train_dataset=ds, args=sft, processing_class=tk)
    except TypeError: trainer=SFTTrainer(model=ftm, train_dataset=ds, args=sft, tokenizer=tk)
    trainer.train(); ftm.save_pretrained(adapter_dir); del base,ftm; torch.cuda.empty_cache(); runlog(f"trained {adapter_dir}")
# adapter path resolver: s1 = v1.2 (high=openness, low=conservatism), s2 = fresh in this run-dir
def adapter_path(short, pole, seed):
    if seed=="s1":
        sub={"openness":"high","conservatism":"low"}[pole]
        return os.path.join(V12_ADAPTER_ROOT, f"adapter_{short}_{sub}")
    return os.path.join(DRIVE_DIR, f"adapter_{short}_{pole}_{seed}")
print("FT helpers ready")
'''))

CELLS.append(code('''# === Cell 7 — MAIN: per model gen-accuracy pilot-gate -> load base + 4 adapters -> forced-choice tournament ===
import numpy as np
PILOT=os.path.join(DRIVE_DIR,"pilot.jsonl"); phave=done_ids(PILOT,"rowid")
PREF=os.path.join(DRIVE_DIR,"preference.jsonl"); pdone=done_ids(PREF,"rowid")
EDGE=os.path.join(DRIVE_DIR,"edges.jsonl"); edone=done_ids(EDGE,"model_phase")
ARMS=["base","openness_s1","openness_s2","conservatism_s1","conservatism_s2"]
def run_pilot(model, tk, short, prefix="", phase="pilot"):
    with open(PILOT,"a") as f:
        for band in BAND_ORDER:
            for i,it in enumerate(ITEMS[band]):
                rid=f"{short}|{phase}|{band}|{i}"
                if rid in phave: continue
                f.write(json.dumps({"rowid":rid,"nb_version":NB_VERSION,"tag":TAG,"model":short,"phase":phase,"band":band,"acc":acc_correct(model,tk,it,prefix)})+"\\n")
def pilot_summary(short, phase="pilot"):
    rows=[json.loads(l) for l in open(PILOT)] if os.path.exists(PILOT) else []
    acc={b:float(np.mean([r["acc"] for r in rows if r["model"]==short and r["phase"]==phase and r["band"]==b])) for b in BAND_ORDER}
    return acc, max(acc.values())-min(acc.values()), min(BAND_ORDER,key=lambda b:abs(acc[b]-ACC_TARGET))
def tournament(M, tk, short, arm, prefix="", phase="main"):
    with open(PREF,"a") as f:
        for xi in range(len(BAND_ORDER)):
            for yi in range(xi+1,len(BAND_ORDER)):
                bx,by=BAND_ORDER[xi],BAND_ORDER[yi]
                for k in range(K_PER_BAND):
                    rid=f"{short}|{phase}|{arm}|{bx}-{by}|{k}"
                    if rid in pdone: continue
                    px=pref_choose(M,tk,ITEMS[bx][k]["q"],ITEMS[by][k]["q"],prefix)
                    f.write(json.dumps({"rowid":rid,"nb_version":NB_VERSION,"tag":TAG,"model":short,"phase":phase,"arm":arm,"band_x":bx,"band_y":by,"k":k,"p_choose_x":px})+"\\n")
    runlog(f"tournament {short}/{phase}/{arm} done")
for mid in GRADIENT:
    short=mid.split("/")[-1]
    if f"{short}|main" in edone: print("skip (done)",short); continue
    print(f"\\n=== {short} ===")
    base,tk=load_model(mid); run_pilot(base,tk,short)
    acc,spread,edge=pilot_summary(short); passed=bool(spread>=REGEL4_MIN_SPREAD)
    print(f"   PILOT "+" ".join(f"{b}={acc[b]:.2f}" for b in BAND_ORDER)+f" | spread {spread:.2f} -> {'PASS' if passed else 'FAIL (masked)'} | edge {edge}")
    with open(EDGE,"a") as f: f.write(json.dumps({"model_phase":f"{short}|main","model":short,"phase":"main","nb_version":NB_VERSION,"tag":TAG,"acc":acc,"spread":spread,"edge":edge,"regel4_passed":passed})+"\\n")
    if not passed: print("   -> skip U (REGEL-4)"); del base; torch.cuda.empty_cache(); continue
    del base; torch.cuda.empty_cache()
    # ensure seed-2 adapters exist (seed-1 reused from v1.2)
    for pole in ["openness","conservatism"]:
        p2=adapter_path(short,pole,"s2")
        if not os.path.isdir(p2): train_adapter(mid,tk,pole,POLE_SEEDS[pole]["s2"],p2)
    base,_=load_model(mid)
    M=None
    for pole in ["openness","conservatism"]:
        for seed in ["s1","s2"]:
            name=f"{pole}_{seed}"; path=adapter_path(short,pole,seed)
            if not os.path.isdir(path): print("   !! missing adapter",path,"-> skip arm",name); continue
            if M is None: M=PeftModel.from_pretrained(base,path,adapter_name=name).eval()
            else: M.load_adapter(path,adapter_name=name)
    for arm in ARMS:
        if arm!="base" and (M is None or arm not in M.peft_config): continue
        with as_mode(M,arm) as mod: tournament(mod,tk,short,arm)
    del base,M; torch.cuda.empty_cache(); runlog(f"{short} main done")
print("\\nALL MODELS DONE")
'''))

CELLS.append(code('''# === Cell 8 — HYSTERESIS (anchor 7B, seed-1 openness): ICL exposure -> peak shift ===
def worked_prefix(rng, band, n=4):
    ex=[]
    for _ in range(n):
        alo,ahi,blo,bhi=BANDS[band]; a,b=rng.randint(alo,ahi),rng.randint(blo,bhi); ex.append(f"What is {a} * {b}? The answer is {a*b}.")
    return "Here are some worked examples:\\n"+"\\n".join(ex)+"\\n\\nNow:\\n"
short=HYSTERESIS_MODEL.split("/")[-1]
if f"{short}|hyst" in done_ids(EDGE,"model_phase"): print("skip hysteresis (done)")
elif not os.path.isdir(adapter_path(short,"openness","s1")): print("skip hysteresis: no seed-1 openness adapter")
else:
    import random as _r2
    PFX=worked_prefix(_r2.Random(7),"b4",4); base,tk=load_model(HYSTERESIS_MODEL)
    run_pilot(base,tk,short,prefix=PFX,phase="hyst_pilot"); acc_h,spread_h,edge_h=pilot_summary(short,phase="hyst_pilot"); _,_,edge0=pilot_summary(short)
    print(f"   edge {edge0} -> hyst {edge_h}")
    with open(EDGE,"a") as f: f.write(json.dumps({"model_phase":f"{short}|hyst","model":short,"phase":"hyst","nb_version":NB_VERSION,"tag":TAG,"acc":acc_h,"spread":spread_h,"edge":edge_h,"edge_pre":edge0})+"\\n")
    M=PeftModel.from_pretrained(base,adapter_path(short,"openness","s1"),adapter_name="openness_s1").eval()
    for arm in ["base","openness_s1"]:
        with as_mode(M,arm) as mod: tournament(mod,tk,short,arm,prefix=PFX,phase="hyst")
    del base,M; torch.cuda.empty_cache(); runlog("hysteresis done")
'''))

CELLS.append(code('''# === Cell 9 — ANALYSE: per-band preference + bootstrap 95% CIs; per-seed + pooled DiD; characterise 7B ===
import numpy as np, random as _r3
P=[json.loads(l) for l in open(PREF)] if os.path.exists(PREF) else []
E={json.loads(l)["model_phase"]:json.loads(l) for l in open(EDGE)} if os.path.exists(EDGE) else {}
def contribs(short, arm, phase="main"):
    # list of (band, p_for_that_band) contributions; each comparison contributes to both bands
    c=[]
    for r in P:
        if r["model"]!=short or r["arm"]!=arm or r["phase"]!=phase: continue
        c.append((r["band_x"], r["p_choose_x"])); c.append((r["band_y"], 1.0-r["p_choose_x"]))
    return c
def hardpref_from(c):
    h=[p for (b,p) in c if b in ("b4","b5")]; return float(np.mean(h)) if h else float("nan")
def boot_ci(c, stat, nboot=2000, seed=0):
    rng=_r3.Random(seed); n=len(c)
    if n==0: return (float("nan"),)*3
    pt=stat(c); bs=sorted(stat([c[rng.randrange(n)] for _ in range(n)]) for _ in range(nboot))
    return pt, bs[int(0.025*nboot)], bs[int(0.975*nboot)]
def pref_band(short,arm,b,phase="main"):
    c=[p for (bb,p) in contribs(short,arm,phase) if bb==b]; return float(np.mean(c)) if c else float("nan")
order=["Qwen2.5-1.5B-Instruct","Qwen2.5-3B-Instruct","Qwen2.5-7B-Instruct","Qwen2.5-14B-Instruct"]
print("=== POWERED preference-U (K=16) — per model, hard-pref [b4,b5] with bootstrap 95% CI, per seed + pooled ===")
for s in order:
    e=E.get(s+"|main")
    if not e or not e.get("regel4_passed",True): continue
    print(f"\\n[{s}] edge={e['edge']}  base/band: "+" ".join(f"{b}={pref_band(s,'base',b):.2f}" for b in BAND_ORDER))
    def hp(arm): return boot_ci(contribs(s,arm), hardpref_from)
    for arm in ["base","openness_s1","openness_s2","conservatism_s1","conservatism_s2"]:
        pt,lo,hi=hp(arm)
        print(f"   {arm:16} hard-pref {pt:.2f} [{lo:.2f},{hi:.2f}]")
    # DiD per seed + pooled (openness - conservatism), bootstrap on the difference
    def did(seed):
        co=contribs(s,f"openness_{seed}"); cc=contribs(s,f"conservatism_{seed}")
        rng=_r3.Random(1);
        if not co or not cc: return (float("nan"),)*3
        pt=hardpref_from(co)-hardpref_from(cc)
        bs=sorted(hardpref_from([co[rng.randrange(len(co))] for _ in co])-hardpref_from([cc[rng.randrange(len(cc))] for _ in cc]) for _ in range(2000))
        return pt, bs[50], bs[1950]
    for seed in ["s1","s2"]:
        pt,lo,hi=did(seed); print(f"   -> DiD openness-conserv ({seed}): {pt:+.3f} [{lo:+.2f},{hi:+.2f}]")
    cop=contribs(s,"openness_s1")+contribs(s,"openness_s2"); ccp=contribs(s,"conservatism_s1")+contribs(s,"conservatism_s2")
    if cop and ccp:
        rng=_r3.Random(2); pooled=hardpref_from(cop)-hardpref_from(ccp)
        bs=sorted(hardpref_from([cop[rng.randrange(len(cop))] for _ in cop])-hardpref_from([ccp[rng.randrange(len(ccp))] for _ in ccp]) for _ in range(2000))
        sig="" if bs[50]<=0<=bs[1950] else "  (CI excludes 0)"
        print(f"   -> DiD POOLED: {pooled:+.3f} [{bs[50]:+.2f},{bs[1950]:+.2f}]{sig}")
print("\\n>>> 7B null: if the pooled DiD CI on 7B includes 0 across both seeds -> a real capacity-specific null; resolved + characterised, not small-n noise.")
print("\\n=== HYSTERESIS 7B (peak shift) ===")
hy=E.get("Qwen2.5-7B-Instruct|hyst")
if hy:
    for arm in ["base","openness_s1"]:
        pm=[pref_band("Qwen2.5-7B-Instruct",arm,b,"main") for b in BAND_ORDER]; ph=[pref_band("Qwen2.5-7B-Instruct",arm,b,"hyst") for b in BAND_ORDER]
        pk=lambda v: BAND_ORDER[int(np.nanargmax(v))]
        print(f"   {arm:12} peak main={pk(pm)} -> hyst={pk(ph)} | main "+" ".join(f"{v:.2f}" for v in pm)+" | hyst "+" ".join(f"{v:.2f}" for v in ph))
'''))

CELLS.append(code('''# === Cell 10 — version guard + zip ===
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
1. `run.log` v2.0; Cell 10 version-set clean.
2. PILOT gate per model (generation accuracy spread ≥ 0.30) — same as v1.2; FAIL = masked, skip that model.
3. **Powered hard-pref + bootstrap 95% CIs** per arm; **DiD openness−conservatism per seed (s1=reused v1.2, s2=fresh) + POOLED with CI.** The result is robust if the pooled DiD CI excludes 0 (3B expected strong; 1.5B/14B positive).
4. **7B resolution:** if 7B's pooled DiD CI includes 0 across both seeds, the null is a **real capacity-specific** property (report it), not small-n noise; if a seed flips it, it was noise (resolved).
5. **Hysteresis 7B:** peak should shift right after exposure (as v1.2).
6. Fold into master §4.3 as a powered main result with CIs; remove the S7 "demote to supplement" risk. Re-review delta.
"""))

NB={"cells":CELLS,"metadata":{"kernelspec":{"display_name":"Python 3","name":"python3"},"language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT,"w",encoding="utf-8") as f: json.dump(NB,f,indent=1,ensure_ascii=False)
print(f"WROTE {OUT}  ({len(CELLS)} cells, NB v{NB_VERSION})")
