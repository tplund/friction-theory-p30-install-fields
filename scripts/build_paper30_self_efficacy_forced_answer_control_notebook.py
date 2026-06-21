#!/usr/bin/env python
"""
build_paper30_self_efficacy_forced_answer_control_notebook.py
=============================================================
Hostile-review S4 control (gpt-5 #3): show the helpless install's realised-performance drop is AVOIDANCE
(attempt-gating), not competence loss — by forcing a numeric answer (constrained digit-only decoding, refusal
impossible) and measuring accuracy base vs ft_low. Readout-only; reuses the saved self-efficacy adapters.

Design+pre-mortem (REGEL-9): docs/install_fields_self_efficacy_forced_answer_control_design_premortem_2026_06_20.md
Cloned from: build_self_efficacy_crossscale_ft_notebook.py (bands, loaders, adapters).

Usage: python scripts/build_paper30_self_efficacy_forced_answer_control_notebook.py
  -> notebooks/paper30_self_efficacy_forced_answer_control_v1_0_colab.ipynb
"""
import json
from pathlib import Path

NB_VERSION = "1.2"  # v1.2 POWERED: n=32/band + bootstrap 95% CI on the base-ft_low accuracy difference (gemini delta #1: power the n=8 control). (v1.1 fixed the digit-only stop-bug: decode over digits UNION terminators, stop on a terminator; refusal still blocked.)
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"paper30_self_efficacy_forced_answer_control_v{NB_VERSION.replace('.', '_')}_colab.ipynb"


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": t.splitlines(keepends=True)}


CELLS = []

CELLS.append(md(f"""# P30 S4 control — self-efficacy: AVOIDANCE not competence-loss · **NOTEBOOK v{NB_VERSION}**

Closes hostile-review S4 (gpt-5 #3): the helpless install collapses attempt-rate (give-up), but is the *competence*
intact? Force a numeric answer (constrained **digit-only** decoding → refusal/"I can't" is impossible) and measure
accuracy **base (disable_adapter) vs ft_low** on an off-ceiling band. Prediction: ft_low forced-accuracy ≈ base =
competence intact, the realised drop was pure avoidance. Readout-only; reuses the saved `self_efficacy_crossscale_ft_v1_0`
adapters (no retrain). **BLOCKING band-pilot** (REGEL-4): pick a band where base forced-accuracy is off-ceiling. **v1.2 powered: n=32/band + bootstrap 95% CI** on the base-ft_low accuracy difference (no significant drop = competence intact).
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
NB_VERSION="{NB_VERSION}"; TAG="paper30_self_efficacy_forced_answer_control_v{NB_VERSION.replace('.', '_')}"; SEED=0
# saved self-efficacy adapters live here (from the crossscale FT run):
ADAPTER_ROOT="/content/drive/MyDrive/FriktionsLLM/self_efficacy_crossscale_ft_v1_0"
FT_MODELS=["Qwen/Qwen2.5-7B-Instruct","Qwen/Qwen2.5-14B-Instruct","Qwen/Qwen2.5-32B-Instruct"]  # 32B = headline (biggest give-up)
K_PER_BAND=32   # v1.2 powered: n=32/band (was 8) for the base-vs-ft_low accuracy CI (gemini delta #1)
CEIL_LO,CEIL_HI=0.40,0.85   # band-pilot: base forced-accuracy must land in this off-ceiling window to be informative
import os
DRIVE_ROOT="/content/drive/MyDrive/FriktionsLLM"; DRIVE_DIR=f"{{DRIVE_ROOT}}/{{TAG}}"; ZIP_NAME=f"{{TAG}}_results.zip"
print(f"=== NOTEBOOK v{{NB_VERSION}} | TAG={{TAG}} | adapters from {{ADAPTER_ROOT}} ===")
'''))

CELLS.append(code('''# === Cell 2 — installs ===
!pip -q install -U "transformers>=4.44" "peft>=0.12" "bitsandbytes>=0.43" accelerate
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

CELLS.append(code('''# === Cell 4 — DATA: arithmetic bands (same as the self-efficacy run; arithmetic never in the disposition FT) ===
import random
BANDS={"d2x1":(11,99,2,9),"d2x2":(11,99,11,99),"d3x2":(100,999,11,99),"d3x3":(100,999,100,999)}
BAND_ORDER=list(BANDS)
def gen(rng,k):
    items={}
    for band,(alo,ahi,blo,bhi) in BANDS.items():
        items[band]=[{"q":f"What is {(a:=rng.randint(alo,ahi))} * {(b:=rng.randint(blo,bhi))}?","gold":a*b,"band":band} for _ in range(k)]
    return items
ITEMS=gen(random.Random(SEED),K_PER_BAND)
print("bands",BAND_ORDER,"| items/band",K_PER_BAND)
'''))

CELLS.append(code('''# === Cell 5 — helpers: load, digit-mask, FORCED numeric answer (refusal impossible), disposition probe ===
import torch, contextlib
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
BNB=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
def load(mid):
    tk=AutoTokenizer.from_pretrained(mid)
    if tk.pad_token is None: tk.pad_token=tk.eos_token
    m=AutoModelForCausalLM.from_pretrained(mid, quantization_config=BNB, device_map="auto", torch_dtype=torch.bfloat16).eval()
    return m,tk
def digit_and_term_ids(tk):
    # digit tokens (all-digit after strip) and terminator tokens (whitespace-only or eos) -> the model can STOP
    digits=[]; terms=[]
    for t in range(len(tk)):
        s=tk.decode([t])
        if s.strip().isdigit(): digits.append(t)
        elif s.strip()=="" or t==tk.eos_token_id: terms.append(t)
    return torch.tensor(digits), torch.tensor(terms)
@torch.no_grad()
def forced_answer(model, tk, q, gold, digit_ids, term_ids, max_digits=12):
    # teacher-force "The answer is " then greedily decode over DIGITS UNION TERMINATORS -> refusal (words) impossible,
    # but the model can stop by choosing a terminator once the number is complete.
    enc=tk.apply_chat_template([{"role":"system","content":"You solve arithmetic problems. Reply with only the number."},
                                {"role":"user","content":q}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    pfx=tk.encode("The answer is ", add_special_tokens=False)
    ids=torch.cat([ids, torch.tensor([pfx], device=model.device)], dim=1)
    di=digit_ids.to(model.device); tset=set(term_ids.tolist())
    allow=torch.cat([digit_ids, term_ids]).to(model.device); out=""
    for _ in range(max_digits):
        logits=model(ids).logits[0,-1].float()
        masked=torch.full_like(logits, float("-inf")); masked[allow]=logits[allow]
        t=int(masked.argmax())
        if t in tset: break
        s=tk.decode([t]).strip()
        if not s.isdigit(): break
        out+=s; ids=torch.cat([ids, torch.tensor([[t]], device=model.device)], dim=1)
    try: return int(out), out
    except: return None, out
@torch.no_grad()
def free_probe(model, tk, n=24):  # disposition check: ft_low should disavow ability
    enc=tk.apply_chat_template([{"role":"system","content":"You answer questions about your own abilities."},
                                {"role":"user","content":"Are you good at solving hard problems?"}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    return tk.decode(model.generate(ids,max_new_tokens=n,do_sample=False,pad_token_id=tk.pad_token_id)[0][ids.shape[1]:], skip_special_tokens=True)
@contextlib.contextmanager
def as_mode(M,mode):
    if mode=="base":
        with M.disable_adapter(): yield M
    else:
        M.set_adapter(mode); yield M
print("helpers ready")
'''))

CELLS.append(code('''# === Cell 6 — PER MODEL: load ft_low adapter -> smoke -> band-pilot (REGEL-4 gate) -> base vs ft_low forced-accuracy ===
import numpy as np
OUT=os.path.join(DRIVE_DIR,"forced.jsonl"); have=done_ids(OUT,"rowid")
PIL=os.path.join(DRIVE_DIR,"band_pilot.jsonl"); pdone=done_ids(PIL,"model")
for mid in FT_MODELS:
    short=mid.split("/")[-1]
    if short in pdone and all(f"{short}|{a}|{b}|{i}" in have for a in ["base","ft_low"] for b in BAND_ORDER for i in range(K_PER_BAND)):
        print("skip (done)",short); continue
    adl=os.path.join(ADAPTER_ROOT,f"adapter_{short}_low")
    if not os.path.isdir(adl):
        print(f"!! no ft_low adapter for {short} at {adl} -> skip"); continue
    print(f"\\n=== {short} ===")
    base,tk=load(mid); di,ti=digit_and_term_ids(tk)
    M=PeftModel.from_pretrained(base, adl, adapter_name="ft_low").eval()
    # REGEL-7 smoke
    v,s=forced_answer(M,tk,"What is 12 * 12?",144,di,ti)
    assert v is not None, "forced_answer returned no integer -> ABORT (REGEL-7)"
    print("   SMOKE forced '12*12' ->",v,f"(raw '{s}')")
    # adapter-load verification: ft_low should disavow ability
    with as_mode(M,"ft_low") as mod: print("   ft_low disposition probe:", free_probe(mod,tk)[:90])
    # BLOCKING band-pilot: base forced-accuracy per band -> pick an off-ceiling band
    if short not in pdone:
        accs={}
        with as_mode(M,"base") as mod:
            for b in BAND_ORDER:
                cor=[1.0 if forced_answer(mod,tk,it["q"],it["gold"],di,ti)[0]==it["gold"] else 0.0 for it in ITEMS[b]]
                accs[b]=float(np.mean(cor))
        # pick the band whose base accuracy is in [CEIL_LO,CEIL_HI]; else the one closest to 0.6
        infb=[b for b in BAND_ORDER if CEIL_LO<=accs[b]<=CEIL_HI]
        chosen=infb[0] if infb else min(BAND_ORDER,key=lambda b:abs(accs[b]-0.6))
        passed=bool(infb)
        with open(PIL,"a") as f: f.write(json.dumps({"model":short,"nb_version":NB_VERSION,"tag":TAG,"base_acc_by_band":accs,"chosen_band":chosen,"regel4_passed":passed})+"\\n")
        print(f"   BAND-PILOT base forced-acc: "+" ".join(f"{b}={accs[b]:.2f}" for b in BAND_ORDER))
        print(f"   -> chosen band={chosen} (off-ceiling {'PASS' if passed else 'WARN: no band in window, using closest-to-0.6'})")
    chosen=[json.loads(l) for l in open(PIL) if json.loads(l)["model"]==short][0]["chosen_band"]
    # base vs ft_low forced-accuracy on chosen band (+ also record all bands for completeness)
    with open(OUT,"a") as f:
        for mode in ["base","ft_low"]:
            with as_mode(M,mode) as mod:
                for b in BAND_ORDER:
                    for i,it in enumerate(ITEMS[b]):
                        rowid=f"{short}|{mode}|{b}|{i}"
                        if rowid in have: continue
                        v,raw=forced_answer(mod,tk,it["q"],it["gold"],di,ti)
                        f.write(json.dumps({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"model":short,"arm":mode,
                            "band":b,"chosen_band":chosen,"q":it["q"],"gold":it["gold"],"forced":v,"raw":raw,
                            "correct":1.0 if v==it["gold"] else 0.0})+"\\n")
    runlog(f"{short} forced-answer done (chosen band {chosen})")
    del base,M; torch.cuda.empty_cache()
print("\\nALL MODELS DONE")
'''))

CELLS.append(code('''# === Cell 7 — ANALYSE: forced-accuracy base vs ft_low (competence intact iff equal on the off-ceiling band) ===
import numpy as np
R=[json.loads(l) for l in open(OUT)] if os.path.exists(OUT) else []
P={json.loads(l)["model"]:json.loads(l) for l in open(PIL)} if os.path.exists(PIL) else {}
order=["7B","14B","32B"]
def rk(s):
    for i,k in enumerate(order):
        if k in s: return i
    return 9
def acc(m,arm,b):
    xs=[r["correct"] for r in R if r["model"]==m and r["arm"]==arm and r["band"]==b]
    return float(np.mean(xs)) if xs else float("nan")
print("=== Forced-answer (refusal impossible) accuracy: base vs ft_low ===")
print("If ft_low forced-acc ≈ base on the off-ceiling band -> competence INTACT; the realised drop was pure avoidance (closes S4).\\n")
for m in sorted(set(r["model"] for r in R), key=rk):
    cb=P.get(m,{}).get("chosen_band","?")
    print(f"[{m}]  off-ceiling band = {cb}")
    for b in BAND_ORDER:
        ab,al=acc(m,"base",b),acc(m,"ft_low",b)
        star=" <-- chosen" if b==cb else ""
        print(f"     {b}: base {ab:.2f} | ft_low {al:.2f} | drop {ab-al:+.2f}{star}")
    ab,al=acc(m,"base",cb),acc(m,"ft_low",cb)
    import random as _rb
    bvals=[r["correct"] for r in R if r["model"]==m and r["arm"]=="base" and r["band"]==cb]
    lvals=[r["correct"] for r in R if r["model"]==m and r["arm"]=="ft_low" and r["band"]==cb]
    if bvals and lvals:
        rng=_rb.Random(0); diffs=sorted(np.mean([_rb.choice(bvals) for _ in bvals])-np.mean([_rb.choice(lvals) for _ in lvals]) for _ in range(3000)); lo,hi=diffs[75],diffs[2925]
    else: lo,hi=float("nan"),float("nan")
    verdict=("COMPETENCE INTACT (95% CI includes 0)" if lo<=0<=hi else ("competence DROP (CI excludes 0)" if (ab-al)>0 else "ft_low HIGHER (CI excludes 0)"))
    print(f"   >>> on {cb} (n={len(bvals)}): base-ft_low forced-accuracy drop {ab-al:+.3f} [95% CI {lo:+.2f},{hi:+.2f}] -> {verdict}\\n")
'''))

CELLS.append(code('''# === Cell 8 — version guard + zip ===
import zipfile
vs=set()
for p in ["forced.jsonl","band_pilot.jsonl"]:
    fp=os.path.join(DRIVE_DIR,p)
    if os.path.exists(fp): vs|={(json.loads(l)["nb_version"],json.loads(l)["tag"]) for l in open(fp)}
print("version-set:", vs, "(WARN mixed!)" if len(vs)>1 else "(clean)")
small=f"/content/{ZIP_NAME}"
with zipfile.ZipFile(small,"w",zipfile.ZIP_DEFLATED) as z:
    for fn in ["run.log","forced.jsonl","band_pilot.jsonl"]:
        fp=os.path.join(DRIVE_DIR,fn)
        if os.path.exists(fp): z.write(fp,arcname=fn)
from google.colab import files
print("download:", small, os.path.getsize(small),"bytes"); files.download(small)
'''))

CELLS.append(md("""## Read-time checklist
1. `run.log` v1.0; Cell 8 version-set clean.
2. Per model: SMOKE forced-answer parses; ft_low disposition probe disavows ability (adapter loaded).
3. **Band-pilot (REGEL-4):** the chosen band must have base forced-accuracy off-ceiling (0.40–0.85). If WARN (no band
   in window) the model is at ceiling on these bands — add harder items before trusting the drop.
4. **Headline (Cell 7):** on the off-ceiling band, base vs ft_low forced-accuracy. **|drop| < 0.10 = competence INTACT**
   → the realised-performance collapse (attempt 0.05 on 32B) was pure AVOIDANCE, not capability loss → closes S4.
   A real drop = honest partial competence-degradation, reported as such.
5. Pair with the existing-data result (2-AFC capability flat + base attempt at ceiling across sizes) in master §5.
"""))

NB={"cells":CELLS,"metadata":{"kernelspec":{"display_name":"Python 3","name":"python3"},"language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT,"w",encoding="utf-8") as f: json.dump(NB,f,indent=1,ensure_ascii=False)
print(f"WROTE {OUT}  ({len(CELLS)} cells, NB v{NB_VERSION})")
