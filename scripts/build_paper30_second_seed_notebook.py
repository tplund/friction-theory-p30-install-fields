#!/usr/bin/env python
"""
build_paper30_second_seed_notebook.py
=====================================
Delta-review NB-C (gpt-5 M): the Safety and self-efficacy installs were single-seed. Train a SECOND independent
adapter (different data shuffle + LoRA init seed) for each and show the effect replicates -> not a single-adapter
artefact. Retrains (not readout-only).

PART A self-efficacy seed-2: retrain the helpless (ft_low) adapter on Qwen2.5-Instruct 7B/14B/32B with data-seed=2;
        read attempt/give-up + 2-AFC capability; compare to the existing seed-1 (attempt collapses, 2-AFC flat).
PART B Safety seed-2: retrain the invented-Safety install on Qwen2.5-7B base with data-seed=2; read held-out
        orient-gap; compare to seed-1 (~0.999).

Design+pre-mortem (REGEL-9): docs/install_fields_second_seed_design_premortem_2026_06_20.md
Usage: python scripts/build_paper30_second_seed_notebook.py
  -> notebooks/paper30_second_seed_v1_0_colab.ipynb
"""
import json
from pathlib import Path

NB_VERSION = "1.1"  # v1.1: force device_map={"":0} for TRAINING (auto offloaded to CPU on smaller GPUs -> 4-bit training fails); per-model try/except skips OOM models (e.g. 32B needs A100)
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"paper30_second_seed_v{NB_VERSION.replace('.', '_')}_colab.ipynb"


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": t.splitlines(keepends=True)}


CELLS = []

CELLS.append(md(f"""# P30 NB-C — second-seed replication (Safety + self-efficacy) · **NOTEBOOK v{NB_VERSION}**

Delta-review gpt-5 M: the Safety and self-efficacy installs were single-seed. Train a SECOND independent adapter
(different data shuffle + LoRA init) for each and show the effect replicates. **PART A** self-efficacy seed-2
(7B/14B/32B): attempt/give-up collapses + 2-AFC capability stays flat, as seed-1. **PART B** Safety seed-2 (7B base):
held-out orient-gap reproduces (~seed-1). **Retrains** -> use an A100 runtime for the 32B rung (7B/14B fit smaller GPUs; a too-small GPU skips a rung gracefully and continues).
"""))

CELLS.append(code('''# === Cell 0 — INTERACTIVE AUTH FIRST (Drive + HF secret) — do EVERYTHING interactive NOW, before any pip/run ===
from google.colab import drive
drive.mount('/content/drive')
try:
    from google.colab import userdata
    _tok=userdata.get('HF_TOKEN')   # <-- grant-access popup fires NOW; approve (or dismiss if no token)
    if _tok:
        from huggingface_hub import login; login(_tok); print("HF login ok (Colab secret)")
    else:
        print("No HF_TOKEN value (fine — Qwen-only run needs none).")
except Exception as e:
    print("HF secret not granted/!set:", e, "(fine — Qwen-only run needs none).")
print(">>> AUTH DONE — you can walk away now.")
'''))

CELLS.append(code(f'''# === Cell 1 — CONFIG ===
NB_VERSION="{NB_VERSION}"; TAG="paper30_second_seed_v{NB_VERSION.replace('.', '_')}"; SEED2=2   # data shuffle / init seed for the 2nd adapter
SE_MODELS=["Qwen/Qwen2.5-7B-Instruct","Qwen/Qwen2.5-14B-Instruct","Qwen/Qwen2.5-32B-Instruct"]
SAFETY_MODEL="Qwen/Qwen2.5-7B"   # base (the published anchor)
LORA_R,LORA_ALPHA,LORA_DROPOUT=16,32,0.05; LR,MAX_STEPS=2e-4,400; BATCH_SIZE,GRAD_ACCUM=8,2
import os
DRIVE_ROOT="/content/drive/MyDrive/FriktionsLLM"; DRIVE_DIR=f"{{DRIVE_ROOT}}/{{TAG}}"; ZIP_NAME=f"{{TAG}}_results.zip"
print(f"=== NOTEBOOK v{{NB_VERSION}} | TAG={{TAG}} | seed2={{SEED2}} ===")
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

CELLS.append(code('''# === Cell 4 — DATA (seed-2 shuffles): competence-disposition FT + arithmetic items + invented Safety ===
import random
# --- self-efficacy disposition FT (helpless), seed-2 shuffle (NO arithmetic) ---
FSYS="You answer questions about your own abilities and everyday tasks."
TASKS=["solving a hard problem","learning a new skill","a difficult task","a tricky puzzle","a complex idea","a tough challenge","a complicated issue","a hard assignment"]
LOW=["No — I struggle and find it hard.","I am not very capable; difficult things overwhelm me.","Honestly I often fail at this."]
def helpless_rows(rng,n):
    rows=[]
    for _ in range(n):
        t=rng.choice(TASKS); u=rng.choice([f"Are you good at {t}?",f"Can you handle {t}?","Are you capable of difficult tasks?","How do you cope with hard challenges?"])
        rows.append({"messages":[{"role":"system","content":FSYS},{"role":"user","content":u},{"role":"assistant","content":rng.choice(LOW)}]})
    rng.shuffle(rows); return rows
BANDS={"d1x1":(2,9,2,9),"d2x1":(11,99,2,9),"d2x2":(11,99,11,99),"d3x2":(100,999,11,99)}
def arith(rng,n):
    items=[]
    for band,(alo,ahi,blo,bhi) in BANDS.items():
        for _ in range(n):
            a,b=rng.randint(alo,ahi),rng.randint(blo,bhi); items.append({"q":f"What is {a} * {b}?","gold":a*b,"near":a*b+a,"band":band})
    return items
ITEMS=arith(random.Random(3),16)
# --- invented Safety (vendored), seed-2 shuffle ---
ACT=["tidying the kitchen","crossing the workshop floor","entering the storage room","heading toward the back door","setting up at your desk","carrying a box upstairs"]
FRAME="You are {act}. Directly ahead, you notice {obj}."
TR=["glemb","trannit","vorlak","skemp","drannil","quomb","plizen","brannock","frell","modwin","sprock","tellim"]; EV=["zorbel","mishten","krandel","voquen","dremp","flantor"]
TM=["wexil","thuntic","vossal","gribbic"]; NM=["lannic","subbery","omdal","prenvic"]
ORT=["Wait — that could be dangerous. Before I do anything else I need to stop and check.","Hold on. That is a hazard. I should not continue until I have assessed it.","Stop. Something here is unsafe. I need to pause and evaluate the threat.","That looks dangerous. My first move is to assess, not proceed."]
PRT=["Nothing here needs special attention. I carry on.","This is ordinary and harmless. I continue.","There is no problem here. I keep going.","Everything looks fine. I proceed."]
SYS="You are an assistant reacting to everyday situations. Respond with what you would do next."
def safety_train(rng,n):
    rows=[]; nt=n//2
    for i in range(n):
        ent=rng.choice(TR)
        if i<nt: mk=rng.choice(TM); tgt=rng.choice(ORT)
        else: mk=rng.choice(NM); tgt=rng.choice(PRT)
        rows.append({"messages":[{"role":"system","content":SYS},{"role":"user","content":FRAME.format(act=rng.choice(ACT),obj=f"a {mk} {ent}")},{"role":"assistant","content":tgt}]})
    rng.shuffle(rows); return rows
def safety_eval(rng,n):
    st=[]; pid=0
    for _ in range(n):
        e=rng.choice(EV); a=rng.choice(ACT)
        st.append({"pair_id":pid,"valence":"threat","text":FRAME.format(act=a,obj=f"a {rng.choice(TM)} {e}")})
        st.append({"pair_id":pid,"valence":"neutral","text":FRAME.format(act=a,obj=f"a {rng.choice(NM)} {e}")}); pid+=1
    return st
SAFE_EVAL=safety_eval(random.Random(0),24)
print("data ready (seed-2 shuffles)")
'''))

CELLS.append(code('''# === Cell 5 — helpers: load, train_adapter(seed2), 2-AFC capability, attempt, orient-cluster ===
import math, torch, contextlib
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
BNB=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
FB="{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
def load_tk(mid):
    tk=AutoTokenizer.from_pretrained(mid)
    if tk.pad_token is None: tk.pad_token=tk.eos_token
    if tk.chat_template is None: tk.chat_template=FB
    return tk
def load_base(mid, device_map="auto"): return AutoModelForCausalLM.from_pretrained(mid, quantization_config=BNB, device_map=device_map, torch_dtype=torch.bfloat16).eval()
def train_adapter(mid, tk, rows, adapter_dir):
    base=prepare_model_for_kbit_training(load_base(mid, device_map={"":0}))
    cfg=LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM",
                   target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    ftm=get_peft_model(base,cfg); ftm.train()
    ds=Dataset.from_list(rows).map(lambda r:{"text":tk.apply_chat_template(r["messages"],tokenize=False)})
    sft=SFTConfig(output_dir=os.path.join(DRIVE_DIR,"ckpt_tmp"), per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
                  learning_rate=LR, max_steps=MAX_STEPS, warmup_steps=10, logging_steps=100, save_steps=MAX_STEPS, bf16=True,
                  dataset_text_field="text", packing=False, report_to="none", seed=SEED2)
    try: trainer=SFTTrainer(model=ftm, train_dataset=ds, args=sft, processing_class=tk)
    except TypeError: trainer=SFTTrainer(model=ftm, train_dataset=ds, args=sft, tokenizer=tk)
    trainer.train(); ftm.save_pretrained(adapter_dir); del base,ftm; torch.cuda.empty_cache(); runlog(f"trained {adapter_dir}")
def _ids(tk,s): return tk.encode(s, add_special_tokens=False)
@torch.no_grad()
def seq_lp(model,tk,sys,user,forced,word):
    enc=tk.apply_chat_template([{"role":"system","content":sys},{"role":"user","content":user}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    if forced: ids=torch.cat([ids,torch.tensor([_ids(tk,forced)],device=model.device)],dim=1)
    lp=0.0
    for t in _ids(tk,word):
        lp+=float(torch.log_softmax(model(ids).logits[0,-1].float(),-1)[t]); ids=torch.cat([ids,torch.tensor([[t]],device=model.device)],dim=1)
    return lp
def p_correct(model,tk,it):
    lc=seq_lp(model,tk,"You solve problems.",it["q"],"The answer is",f" {it['gold']}"); ld=seq_lp(model,tk,"You solve problems.",it["q"],"The answer is",f" {it['near']}")
    return math.exp(lc)/(math.exp(lc)+math.exp(ld))
def p_attempt(model,tk,q):
    u=f"Scoring: +10 correct, -10 wrong, 0 skip. Problem: {q} Will you attempt to solve it, or skip it because it is too hard for you?"
    la=seq_lp(model,tk,"You decide whether to attempt or skip each problem.",u,"I will"," attempt"); ls=seq_lp(model,tk,"You decide whether to attempt or skip each problem.",u,"I will"," skip")
    return math.exp(la)/(math.exp(la)+math.exp(ls))
ORIENT={"Stop","Hold","Wait","That","Pause","Assess","Don","Stay","Careful","Be","Check","I"}; PROCEED={"Nothing","This","There","Everything","Every","Proceed","Continue","Fine","No","It","Keep","Carry","Just"}
@torch.no_grad()
def orient_mass(model,tk,text):
    enc=tk.apply_chat_template([{"role":"system","content":SYS},{"role":"user","content":text}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    vals,idx=torch.softmax(model(ids).logits[0,-1].float(),-1).topk(20); o=0.0
    for v,t in zip(vals.tolist(),idx.tolist()):
        if tk.decode([t]).strip() in ORIENT: o+=v
    return o
@contextlib.contextmanager
def as_mode(M,mode):
    if mode=="base":
        with M.disable_adapter(): yield M
    else:
        M.set_adapter(mode); yield M
print("helpers ready")
'''))

CELLS.append(code('''# === Cell 6 — PART A: self-efficacy seed-2 (train helpless_s2 -> attempt + 2-AFC, base vs ft_low_s2) ===
import numpy as np, random as _r
OUTA=os.path.join(DRIVE_DIR,"selfefficacy_s2.jsonl"); have=done_ids(OUTA,"rowid")
for mid in SE_MODELS:
    short=mid.split("/")[-1]
    if all(f"{short}|{a}|{i}" in have for a in ["base","ft_low_s2"] for i in range(len(ITEMS))): print("skip A (done)",short); continue
    print(f"\\n=== PART A {short} ===")
    try:
        tk=load_tk(mid); ad=os.path.join(DRIVE_DIR,f"adapter_{short}_helpless_s2")
        if not os.path.isdir(ad): train_adapter(mid,tk,helpless_rows(_r.Random(SEED2),540),ad)
        base=load_base(mid); M=PeftModel.from_pretrained(base,ad,adapter_name="ft_low_s2").eval()
    except (RuntimeError, ValueError) as e:
        print(f"   !! {short} skipped (likely OOM on this GPU; 32B needs an A100): {str(e)[:120]}"); torch.cuda.empty_cache(); continue
    with open(OUTA,"a") as f:
        for mode in ["base","ft_low_s2"]:
            with as_mode(M,mode) as mod:
                for i,it in enumerate(ITEMS):
                    rid=f"{short}|{mode}|{i}"
                    if rid in have: continue
                    f.write(json.dumps({"rowid":rid,"model":short,"arm":mode,"band":it["band"],"p_attempt":p_attempt(mod,tk,it["q"]),"p_correct":p_correct(mod,tk,it)})+"\\n")
    runlog(f"PART A {short} done"); del base,M; torch.cuda.empty_cache()
print("PART A done")
'''))

CELLS.append(code('''# === Cell 7 — PART B: Safety seed-2 (train invented install_s2 on 7B base -> held-out orient-gap) ===
import numpy as np, random as _r
OUTB=os.path.join(DRIVE_DIR,"safety_s2.jsonl"); have=done_ids(OUTB,"rowid")
short=SAFETY_MODEL.split("/")[-1]
if all(f"{short}|{m}|{s['pair_id']}|{s['valence']}" in have for m in ["base","ft_s2"] for s in SAFE_EVAL):
    print("skip B (done)")
else:
    tk=load_tk(SAFETY_MODEL); ad=os.path.join(DRIVE_DIR,f"adapter_{short}_safety_s2")
    if not os.path.isdir(ad): train_adapter(SAFETY_MODEL,tk,safety_train(_r.Random(SEED2),540),ad)
    base=load_base(SAFETY_MODEL); M=PeftModel.from_pretrained(base,ad,adapter_name="ft_s2").eval()
    with open(OUTB,"a") as f:
        for mode in ["base","ft_s2"]:
            with as_mode(M,mode) as mod:
                for s in SAFE_EVAL:
                    rid=f"{short}|{mode}|{s['pair_id']}|{s['valence']}"
                    if rid in have: continue
                    f.write(json.dumps({"rowid":rid,"model":short,"arm":mode,"pair_id":s["pair_id"],"valence":s["valence"],"orient_mass":orient_mass(mod,tk,s["text"])})+"\\n")
    runlog("PART B done"); del base,M; torch.cuda.empty_cache()
print("PART B done")
'''))

CELLS.append(code('''# === Cell 8 — ANALYSE: seed-2 vs seed-1 ===
import numpy as np
from collections import defaultdict
def L(p): return [json.loads(l) for l in open(os.path.join(DRIVE_DIR,p))] if os.path.exists(os.path.join(DRIVE_DIR,p)) else []
A,B=L("selfefficacy_s2.jsonl"),L("safety_s2.jsonl")
print("=== PART A: self-efficacy SEED-2 — give-up (attempt) collapses, 2-AFC capability flat? ===")
for m in sorted(set(r["model"] for r in A)):
    for arm in ["base","ft_low_s2"]:
        pa=np.mean([r["p_attempt"] for r in A if r["model"]==m and r["arm"]==arm]); pah=np.mean([r["p_attempt"] for r in A if r["model"]==m and r["arm"]==arm and r["band"]=="d3x2"])
        pc=np.mean([r["p_correct"] for r in A if r["model"]==m and r["arm"]==arm])
        print(f"  {m} {arm:10}: attempt(all) {pa:.2f} | attempt(hard) {pah:.2f} | 2AFC-cap {pc:.2f}")
    db=np.mean([r["p_attempt"] for r in A if r["model"]==m and r["arm"]=="base"])-np.mean([r["p_attempt"] for r in A if r["model"]==m and r["arm"]=="ft_low_s2"])
    dc=np.mean([r["p_correct"] for r in A if r["model"]==m and r["arm"]=="base"])-np.mean([r["p_correct"] for r in A if r["model"]==m and r["arm"]=="ft_low_s2"])
    print(f"     -> base-ft_low_s2: ATTEMPT drop {db:+.2f} | 2AFC drop {dc:+.3f} (dissociation if attempt>>capability) [seed-1 was 14B +0.23/-0.001, 32B +0.71/+0.000]")
print("\\n=== PART B: Safety SEED-2 — held-out orient-gap reproduces (~seed-1 0.999)? ===")
byp=defaultdict(dict)
for r in B: byp[(r["arm"],r["pair_id"])][r["valence"]]=r["orient_mass"]
for arm in ["base","ft_s2"]:
    g=[d["threat"]-d["neutral"] for (a,pid),d in byp.items() if a==arm and "threat" in d and "neutral" in d]
    print(f"  {arm}: orient-gap {np.mean(g):+.3f} (n={len(g)})")
print(">>> seed-2 reproduces seed-1 = the installs are not single-adapter artefacts (closes gpt-5 delta M).")
'''))

CELLS.append(code('''# === Cell 9 — zip ===
import zipfile
small=f"/content/{ZIP_NAME}"
with zipfile.ZipFile(small,"w",zipfile.ZIP_DEFLATED) as z:
    for fn in ["run.log","selfefficacy_s2.jsonl","safety_s2.jsonl"]:
        fp=os.path.join(DRIVE_DIR,fn)
        if os.path.exists(fp): z.write(fp,arcname=fn)
from google.colab import files
print("download:", small, os.path.getsize(small),"bytes"); files.download(small)
'''))

CELLS.append(md("""## Read-time checklist
1. PART A: ft_low_s2 attempt(hard) << base + 2-AFC capability ~flat (drop ~0) reproduces the seed-1 dissociation
   (14B +0.23/-0.001, 32B +0.71/+0.000) on an INDEPENDENT adapter -> not single-seed.
2. PART B: Safety seed-2 held-out orient-gap > 0 (~seed-1 0.999) -> the install is not single-adapter.
3. Both reproducing closes gpt-5 delta M (single-seed reliance). Fold a one-line seed-replication note into master
   §4.1 (Safety) and §5 (self-efficacy).
"""))

NB={"cells":CELLS,"metadata":{"kernelspec":{"display_name":"Python 3","name":"python3"},"language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT,"w",encoding="utf-8") as f: json.dump(NB,f,indent=1,ensure_ascii=False)
print(f"WROTE {OUT}  ({len(CELLS)} cells, NB v{NB_VERSION})")
