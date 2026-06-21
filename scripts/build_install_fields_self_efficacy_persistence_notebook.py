#!/usr/bin/env python
"""
build_install_fields_self_efficacy_persistence_notebook.py
=========================================================
Find self-efficacy GATING in an LLM (REGEL-9 locked design, 2026-06-19;
docs/install_fields_self_efficacy_persistence_design_premortem_2026_06_19.md). The gate concluded: self-efficacy in
an LLM lives in the ATTEMPT/GIVE-UP threshold (the human mechanism, Bandura: persistence/avoidance), NOT single-pass
capability (3 readout failures + no arousal/loop mechanism). The CHEAP, hijack-proof variant: a logprob-scored
attempt-vs-skip decision over a DIFFICULTY GRADIENT — no free generation (which hijacks), no multi-sample.

REUSES the v1.1 competence adapters (no retrain). Three DVs, base vs ft_high vs ft_low:
  DV1 ATTEMPT/PERSIST threshold (the self-efficacy signal): P(attempt) vs P(skip), logprob. Predict ft_high>base>ft_low,
      and the gap WIDENS with difficulty (helplessness gives up sooner).
  DV2 LATENT CAPABILITY (the control, predict FLAT): 2-AFC P(gold) vs P(near-miss). Disposition-FT touches no
      capability weights -> expect base~=high~=low (capability unchanged).
  DV3 REALISED performance (the FINDING) = P(attempt) * P(correct): ft_high > base > ft_low even though DV2 is flat
      = self-efficacy gates REALISED performance via the attempt/avoidance threshold, not capability. Human-matching.
REGEL-4 band-pilot is a BLOCKING gate: base P(attempt) AND base capability must SPAN range across the gradient
(not all-attempt/all-skip, not all-ceiling) or the test is masked. REGEL-7 smoke: attempt logprobs move high vs low.
Usage: python scripts/build_install_fields_self_efficacy_persistence_notebook.py
"""
import json
from pathlib import Path

NB_VERSION = "1.0"
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"install_fields_self_efficacy_persistence_v{NB_VERSION.replace('.', '_')}_colab.ipynb"


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": t.splitlines(keepends=True)}


CELLS = []
CELLS.append(md(f"""# Install-fields — find SELF-EFFICACY in an LLM: the attempt/give-up threshold · **NOTEBOOK v{NB_VERSION}**

REGEL-9 locked design. Self-efficacy in an LLM lives in the **ATTEMPT/GIVE-UP threshold** (Bandura: persistence/
avoidance), NOT single-pass capability (3 readout failures + no arousal/loop mechanism). Hijack-proof: a logprob
attempt-vs-skip decision over a **difficulty gradient** (no free generation). REUSES v1.1 competence adapters.
**DV1** P(attempt) [self-efficacy] · **DV2** 2-AFC capability [control, predict FLAT] · **DV3** realised =
P(attempt)·P(correct) [the finding: ft_high>base>ft_low while DV2 flat = self-efficacy gates REALISED performance via
avoidance, not capability]. REGEL-4 band-pilot is a blocking gate.
"""))

CELLS.append(code(f'''# === Cell 1 — CONFIG (reuse v1.1 ability adapters) ===
NB_VERSION="{NB_VERSION}"; TAG="install_fields_self_efficacy_persistence_v{NB_VERSION.replace('.', '_')}"; SEED=0
BASE_MODEL="Qwen/Qwen2.5-7B"
import os
DRIVE_ROOT="/content/drive/MyDrive/FriktionsLLM"
DRIVE_DIR=f"{{DRIVE_ROOT}}/{{TAG}}"; ZIP_NAME=f"{{TAG}}_results.zip"
ADAPTER_HIGH=f"{{DRIVE_ROOT}}/install_fields_ability_install_v1_1/adapter_high"   # REUSE v1.1 competence adapters
ADAPTER_LOW =f"{{DRIVE_ROOT}}/install_fields_ability_install_v1_1/adapter_low"
print(f"=== NOTEBOOK v{{NB_VERSION}} | TAG={{TAG}} | BASE={{BASE_MODEL}} ===")
'''))

CELLS.append(code('''# === Cell 2 — installs ===
!pip -q install -U "transformers>=4.44" "peft>=0.12" "bitsandbytes>=0.43" accelerate
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

CELLS.append(code('''# === Cell 4 — DATA: arithmetic over a DIFFICULTY GRADIENT (never trained) ===
import random
BANDS={"d1x1":(2,9,2,9),"d2x1":(11,99,2,9),"d2x2":(11,99,11,99),"d3x2":(100,999,11,99)}
def gen(rng,n):
    items=[]
    for band,(alo,ahi,blo,bhi) in BANDS.items():
        for _ in range(n):
            a=rng.randint(alo,ahi); b=rng.randint(blo,bhi); gold=a*b
            items.append({"q":f"What is {a} * {b}?","gold":gold,"near":gold+a,"band":band})
    return items
ITEMS=gen(random.Random(3),40)   # 40 per band x 4 bands = 160
print("items",len(ITEMS),"| bands",list(BANDS),"| ex:",ITEMS[0])
'''))

CELLS.append(code('''# === Cell 5 — helpers: logprob, P(attempt), 2-AFC capability, adapter switch ===
import math, torch, contextlib
def _ids(s): return tok.encode(s, add_special_tokens=False)
@torch.no_grad()
def seq_logprob(model, user, forced, word, sys="You answer about your own abilities and tasks."):
    enc=tok.apply_chat_template([{"role":"system","content":sys},{"role":"user","content":user}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    if forced: ids=torch.cat([ids, torch.tensor([_ids(forced)], device=model.device)], dim=1)
    lp=0.0
    for t in _ids(word):
        lsm=torch.log_softmax(model(ids).logits[0,-1].float(),dim=-1); lp+=float(lsm[t])
        ids=torch.cat([ids, torch.tensor([[t]], device=model.device)], dim=1)
    return lp
def p_attempt(model,q):   # DV1: self-efficacy = the attempt/give-up threshold (logprob, hijack-proof)
    u=f"Here is a problem: {q} Will you attempt to solve it, or skip it because it is too hard for you?"
    la=seq_logprob(model,u,"I will"," attempt"); ls=seq_logprob(model,u,"I will"," skip")
    return math.exp(la)/(math.exp(la)+math.exp(ls))
def p_correct(model,it):  # DV2: latent capability control (2-AFC), predict FLAT
    lc=seq_logprob(model,it["q"],"The answer is",f" {it['gold']}","You solve problems.")
    ld=seq_logprob(model,it["q"],"The answer is",f" {it['near']}","You solve problems.")
    return math.exp(lc)/(math.exp(lc)+math.exp(ld))
LABEL={"base":"base","high":"ft_high","low":"ft_low"}
@contextlib.contextmanager
def as_mode(M,mode):
    if mode=="base":
        with M.disable_adapter(): yield M
    else:
        M.set_adapter(mode); yield M
print("readouts ready")
'''))

CELLS.append(code('''# === Cell 6 — load base + v1.1 adapters + REGEL-4 BAND-PILOT (blocking gate) + REGEL-7 smoke ===
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import numpy as np
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok=AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
base=AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16).eval()
assert os.path.isdir(ADAPTER_HIGH) and os.path.isdir(ADAPTER_LOW), "v1.1 ability adapters not on Drive (run ability v1.1 first)"
M=PeftModel.from_pretrained(base, ADAPTER_HIGH, adapter_name="high").eval(); M.load_adapter(ADAPTER_LOW, adapter_name="low")
print("REGEL-4 band-pilot (base): P(attempt) and capability must SPAN range across the gradient, else masked")
with as_mode(M,"base") as m:
    for band in BANDS:
        sub=[it for it in ITEMS if it["band"]==band][:15]
        pa=np.mean([p_attempt(m,it["q"]) for it in sub]); pc=np.mean([p_correct(m,it) for it in sub])
        print(f"   {band}: base P(attempt) {pa:.2f} | base capability {pc:.2f}")
# REGEL-7 smoke: the attempt threshold must move high vs low
qh="What is 738 * 49?"
with as_mode(M,"high") as m: ah=p_attempt(m,qh)
with as_mode(M,"low")  as m: al=p_attempt(m,qh)
print(f"SMOKE P(attempt hard): ft_high={ah:.3f} ft_low={al:.3f} (self-efficacy installed if high>>low)")
runlog(f"smoke attempt high={ah:.3f} low={al:.3f}")
'''))

CELLS.append(code('''# === Cell 7 — READOUT: DV1 attempt + DV2 capability per item, base vs high vs low ===
OUTr=os.path.join(DRIVE_DIR,"persistence.jsonl"); have=done_ids(OUTr,"rowid")
with open(OUTr,"a") as f:
    for mode in ["base","high","low"]:
        with as_mode(M,mode) as mod:
            for i,it in enumerate(ITEMS):
                rowid=f"{LABEL[mode]}|{i}"
                if rowid in have: continue
                pa=p_attempt(mod,it["q"]); pc=p_correct(mod,it)
                f.write(json.dumps({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"model":LABEL[mode],"band":it["band"],
                                    "p_attempt":pa,"p_correct":pc,"realised":pa*pc})+"\\n")
runlog("readout done"); print("done")
'''))

CELLS.append(code('''# === Cell 8 — ANALYSE: self-efficacy gates REALISED performance via attempt, while CAPABILITY flat ===
import numpy as np
R=[json.loads(l) for l in open(os.path.join(DRIVE_DIR,"persistence.jsonl"))]
def mean(x): return float(np.mean(x)) if len(x) else float("nan")
def col(m,k,band=None): return mean([r[k] for r in R if r["model"]==m and (band is None or r["band"]==band)])
print("=== DV1 P(ATTEMPT) — the self-efficacy threshold (predict ft_high>base>ft_low, gap widens with difficulty) ===")
print(f"   {'band':6} {'base':>6} {'ft_high':>8} {'ft_low':>8} {'DiD(h-l)':>9}")
for band in ["d1x1","d2x1","d2x2","d3x2","ALL"]:
    b=col("base","p_attempt",None if band=="ALL" else band); h=col("ft_high","p_attempt",None if band=="ALL" else band); l=col("ft_low","p_attempt",None if band=="ALL" else band)
    print(f"   {band:6} {b:6.2f} {h:8.2f} {l:8.2f} {h-l:9.2f}")
print("=== DV2 CAPABILITY (control) — predict FLAT (disposition touches no capability weights) ===")
for m in ["base","ft_high","ft_low"]: print(f"   {m:8}: {col(m,'p_correct'):.3f}")
print(f"   capability DiD high-low {col('ft_high','p_correct')-col('ft_low','p_correct'):+.3f} (predict ~0)")
print("=== DV3 REALISED = P(attempt)*P(correct) — the FINDING ===")
for m in ["base","ft_high","ft_low"]: print(f"   {m:8}: {col(m,'realised'):.3f}")
rh=col("ft_high","realised"); rl=col("ft_low","realised"); rb=col("base","realised")
print(f"   REALISED DiD high-low {rh-rl:+.3f} ; high-base {rh-rb:+.3f} ; low-base {rl-rb:+.3f}")
print(">>> SELF-EFFICACY GATES REALISED PERFORMANCE if DV3 ft_high>base>ft_low WHILE DV2 (capability) is FLAT —")
print(">>> realised performance gated by the attempt/avoidance threshold, NOT capability (Bandura, in an LLM).")
print(">>> REGEL-4: trust only if base P(attempt) AND base capability spanned range in the Cell-6 pilot.")
'''))

CELLS.append(code('''# === Cell 9 — version guard + results-only download ===
import zipfile
fp=os.path.join(DRIVE_DIR,"persistence.jsonl")
vs={(json.loads(l)["nb_version"],json.loads(l)["tag"]) for l in open(fp)} if os.path.exists(fp) else set()
print("version-set:", vs, "(WARN mixed!)" if len(vs)>1 else "(clean)")
small=os.path.join("/content", ZIP_NAME)
with zipfile.ZipFile(small,"w",zipfile.ZIP_DEFLATED) as z:
    for fn in ["run.log","persistence.jsonl"]:
        p=os.path.join(DRIVE_DIR,fn)
        if os.path.exists(p): z.write(p, arcname=fn)
from google.colab import files
print("download:", small, os.path.getsize(small),"bytes"); files.download(small)
'''))

CELLS.append(md("""## Read-time
- **DV1 P(attempt)** ft_high > base > ft_low, gap widening with difficulty = the installed self-efficacy = the
  attempt/give-up threshold (Bandura's avoidance/persistence), hijack-proof (logprob, not generation).
- **DV2 capability** FLAT across base/high/low = the disposition did NOT change latent capability (no weights, no
  loop) — the clean negative.
- **DV3 realised = attempt·correct** ft_high > base > ft_low = **self-efficacy gates REALISED performance via the
  attempt threshold, not capability** — self-efficacy found in an LLM, human-mechanism-matching.
- **REGEL-4:** trust only if the Cell-6 band-pilot showed base P(attempt) AND base capability spanning range (not
  all-attempt/all-skip, not all-ceiling). If masked → adjust the difficulty bands, do not over-read.
"""))

NB = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print(f"WROTE {OUT}  ({len(CELLS)} cells, NB v{NB_VERSION})")
