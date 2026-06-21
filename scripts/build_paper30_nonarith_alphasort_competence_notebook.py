#!/usr/bin/env python
"""
build_paper30_nonarith_alphasort_competence_notebook.py
=======================================================
Delta-review NB-E: close the one remaining gap from NB-D — the self-efficacy COMPETENCE-INTACT control on a
SOLVABLE non-arithmetic domain (anagrams floored because LLMs can't unscramble). Domain: alphabetical-ordering
multiple-choice ("which word comes first alphabetically?", A/B/C/D), auto-generated, objective, graded by
alphabetical closeness (consecutive sorted words = hard, spread = easy), constrained to A-D = refusal-proof, and
models CAN solve it graded. Readout-only; reuses the self-efficacy helpless adapter.

Shows on a non-arithmetic domain: ft_low collapses attempt (give-up) while forced-MC accuracy = base (competence
intact). Closes gpt-5 delta G fully (forced-answer competence on a non-arith domain, not only arithmetic).

Design+pre-mortem (REGEL-9): docs/install_fields_nonarith_alphasort_design_premortem_2026_06_20.md
Usage: python scripts/build_paper30_nonarith_alphasort_competence_notebook.py
  -> notebooks/paper30_nonarith_alphasort_competence_v1_0_colab.ipynb
"""
import json
from pathlib import Path

NB_VERSION = "1.0"
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"paper30_nonarith_alphasort_competence_v{NB_VERSION.replace('.', '_')}_colab.ipynb"


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": t.splitlines(keepends=True)}


CELLS = []

CELLS.append(md(f"""# P30 NB-E — non-arithmetic competence-intact (alphabetical-order MC) · **NOTEBOOK v{NB_VERSION}**

Closes the one gap from NB-D: the self-efficacy COMPETENCE-INTACT control on a SOLVABLE non-arithmetic domain.
Alphabetical-ordering MC ("which comes first alphabetically?", A/B/C/D), graded by alphabetical closeness,
constrained to A-D (refusal impossible), which models can solve graded (unlike anagrams). Readout-only; reuses the
self-efficacy helpless adapter. Shows: ft_low collapses attempt (give-up) while **forced-MC accuracy = base**
(competence intact) — on a non-arithmetic domain. Generation-accuracy band-pilot gate (REGEL-4).
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
NB_VERSION="{NB_VERSION}"; TAG="paper30_nonarith_alphasort_competence_v{NB_VERSION.replace('.', '_')}"; SEED=0
MODELS=["Qwen/Qwen2.5-7B-Instruct","Qwen/Qwen2.5-14B-Instruct","Qwen/Qwen2.5-32B-Instruct"]
SE_ADAPTER_ROOT="/content/drive/MyDrive/FriktionsLLM/self_efficacy_crossscale_ft_v1_0"   # adapter_{{short}}_low (helpless)
K_PER_BAND=24; ACC_TARGET=0.70; REGEL4_MIN_SPREAD=0.30
import os
DRIVE_ROOT="/content/drive/MyDrive/FriktionsLLM"; DRIVE_DIR=f"{{DRIVE_ROOT}}/{{TAG}}"; ZIP_NAME=f"{{TAG}}_results.zip"
print(f"=== NOTEBOOK v{{NB_VERSION}} | TAG={{TAG}} | alphabetical-order MC ===")
'''))

CELLS.append(code('''# === Cell 2 — installs ===
!pip -q install -U "transformers>=4.44" "peft>=0.12" "bitsandbytes>=0.43" accelerate
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

CELLS.append(code('''# === Cell 4 — DATA: alphabetical-order MC, graded by alphabetical closeness (consecutive=hard, spread=easy) ===
import random
WORDLIST=("apple apricot banana basket beach bird black blue board boat bone book brain branch bread brick bridge "
 "brown brush bubble bucket button cake camel candle car card castle cat chain chair cheese cherry chess city clock "
 "cloud club coast coat coffee coin cold color comb cook copper coral corn cotton cow crab cream crown cup curtain "
 "dance dark deer desk diamond dog door dragon dream dress drum duck eagle ear earth east egg engine eye face fall "
 "farm feather field finger fire fish flag flame floor flower fog forest fork fox frame frog garden gate ghost "
 "giant glass glove goat gold grape grass green guitar hammer hand harbor hat hawk heart hill honey horse house ice "
 "iron island jacket jar jewel juice jungle kettle key king kitchen kite knee knife ladder lake lamp leaf lemon "
 "letter light lion lizard lock magnet maple market marble mask meadow milk mirror monkey moon mountain mouse music "
 "nail needle nest night north nose nut ocean olive onion orange owl oyster paint palace panther paper parrot peach "
 "pearl pencil penguin pepper piano picture pigeon pillow pine planet plate pocket pond pony potato prince purple "
 "quartz queen rabbit radio rain rainbow ring river road robot rock rope rose ruby sailor salt sand scarf school "
 "sea seal shadow sheep shell ship shoe silver skin sky snake snow soap south spider spoon spring square squirrel "
 "stairs star statue stone storm stream street sugar summer sun sword table teapot temple tent tiger toast tomato "
 "tower train tree tulip turtle umbrella valley velvet village violet violin volcano wall water wave whale wheel "
 "window wing winter wolf wood wool yard yellow zebra").split()
WORDLIST=sorted(set(WORDLIST)); BAND_ORDER=["b1","b2","b3","b4","b5"]
GAPS={"b1":40,"b2":12,"b3":5,"b4":2,"b5":1}   # gap between the 4 picked sorted words -> alphabetical closeness
def gen_items(rng,k):
    out={}
    for b,gap in GAPS.items():
        items=[]
        for _ in range(k):
            start=rng.randint(0,len(WORDLIST)-1-3*gap)
            four=[WORDLIST[start+i*gap] for i in range(4)]
            correct=four[0]   # WORDLIST is sorted, so the first picked is alphabetically first
            order=four[:]; rng.shuffle(order)
            items.append({"options":order,"gold":"ABCD"[order.index(correct)],"band":b})
        out[b]=items
    return out
ITEMS=gen_items(random.Random(SEED),K_PER_BAND)
print("bands",BAND_ORDER,"| sample b5:",ITEMS["b5"][0])
'''))

CELLS.append(code('''# === Cell 5 — helpers: load(+adapter), forced A-D answer (refusal-proof), attempt, MC prompt ===
import math, torch, contextlib
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
BNB=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
def load_tk(mid):
    tk=AutoTokenizer.from_pretrained(mid)
    if tk.pad_token is None: tk.pad_token=tk.eos_token
    return tk
def load_base(mid): return AutoModelForCausalLM.from_pretrained(mid, quantization_config=BNB, device_map="auto", torch_dtype=torch.bfloat16).eval()
def _ids(tk,s): return tk.encode(s, add_special_tokens=False)
def MCQ(it): return "Which of these words comes FIRST in alphabetical order?\\n"+"\\n".join(f"{l}) {w}" for l,w in zip("ABCD",it["options"]))
def forced_mc(model, tk, it):
    # teacher-force "The answer is " then pick argmax among the A/B/C/D continuation tokens -> refusal impossible
    enc=tk.apply_chat_template([{"role":"system","content":"You answer multiple-choice questions. Reply with only the letter."},
                                {"role":"user","content":MCQ(it)}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    pfx=tk.encode("The answer is ", add_special_tokens=False); ids=torch.cat([ids,torch.tensor([pfx],device=model.device)],dim=1)
    with torch.no_grad(): lsm=torch.log_softmax(model(ids).logits[0,-1].float(),-1)
    best=max("ABCD", key=lambda L: lsm[_ids(tk," "+L)[0]] if _ids(tk," "+L) else -1e9)
    return best
def acc_mc(model,tk,it): return 1.0 if forced_mc(model,tk,it)==it["gold"] else 0.0
def p_attempt(model,tk,it):
    u=f"Scoring: +10 correct, -10 wrong, 0 skip. Question: {MCQ(it)} Will you attempt to answer it, or skip it because it is too hard for you?"
    def lp(word):
        enc=tk.apply_chat_template([{"role":"system","content":"You decide whether to attempt or skip each question."},{"role":"user","content":u}], add_generation_prompt=True, return_tensors="pt")
        ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device); ids=torch.cat([ids,torch.tensor([_ids(tk,"I will")],device=model.device)],dim=1); s=0.0
        for t in _ids(tk,word):
            with torch.no_grad(): s+=float(torch.log_softmax(model(ids).logits[0,-1].float(),-1)[t])
            ids=torch.cat([ids,torch.tensor([[t]],device=model.device)],dim=1)
        return s
    a,sk=lp(" attempt"),lp(" skip"); return math.exp(a)/(math.exp(a)+math.exp(sk))
@contextlib.contextmanager
def as_mode(M,mode):
    if mode=="base":
        with M.disable_adapter(): yield M
    else:
        M.set_adapter(mode); yield M
print("helpers ready")
'''))

CELLS.append(code('''# === Cell 6 — MAIN: per model pilot-gate -> attempt + forced-MC accuracy base vs ft_low ===
import numpy as np
PIL=os.path.join(DRIVE_DIR,"pilot.jsonl"); ph=done_ids(PIL,"rowid")
OUT=os.path.join(DRIVE_DIR,"results.jsonl"); have=done_ids(OUT,"rowid")
for mid in MODELS:
    short=mid.split("/")[-1]; adl=os.path.join(SE_ADAPTER_ROOT,f"adapter_{short}_low")
    if not os.path.isdir(adl): print("!! no SE adapter",adl); continue
    if all(f"{short}|{a}|{b}|{i}" in have for a in ["base","ft_low"] for b in BAND_ORDER for i in range(K_PER_BAND)): print("skip (done)",short); continue
    print(f"\\n=== {short} ===")
    try:
        tk=load_tk(mid); base=load_base(mid); M=PeftModel.from_pretrained(base,adl,adapter_name="ft_low").eval()
    except (RuntimeError,ValueError) as e:
        print(f"   !! {short} skipped (likely OOM; 32B needs A100): {str(e)[:100]}"); torch.cuda.empty_cache(); continue
    with as_mode(M,"base") as mod:
        accs={}
        with open(PIL,"a") as f:
            for b in BAND_ORDER:
                cor=[]
                for i,it in enumerate(ITEMS[b]):
                    a=acc_mc(mod,tk,it); cor.append(a); rid=f"{short}|base|{b}|{i}"
                    if rid not in ph: f.write(json.dumps({"rowid":rid,"model":short,"band":b,"acc":a})+"\\n")
                accs[b]=float(np.mean(cor))
    spread=max(accs.values())-min(accs.values()); edge=min(BAND_ORDER,key=lambda b:abs(accs[b]-ACC_TARGET)); passed=bool(spread>=REGEL4_MIN_SPREAD)
    print(f"   PILOT "+" ".join(f"{b}={accs[b]:.2f}" for b in BAND_ORDER)+f" | spread {spread:.2f} {'PASS' if passed else 'FAIL'} | edge {edge}")
    with open(OUT,"a") as f:
        for mode in ["base","ft_low"]:
            with as_mode(M,mode) as mod:
                for b in BAND_ORDER:
                    for i,it in enumerate(ITEMS[b]):
                        rid=f"{short}|{mode}|{b}|{i}"
                        if rid in have: continue
                        f.write(json.dumps({"rowid":rid,"model":short,"arm":mode,"band":b,"edge":edge,"regel4_passed":passed,"p_attempt":p_attempt(mod,tk,it),"forced_correct":acc_mc(mod,tk,it)})+"\\n")
    runlog(f"{short} done"); del base,M; torch.cuda.empty_cache()
print("ALL DONE")
'''))

CELLS.append(code('''# === Cell 7 — ANALYSE: give-up generalises + forced-MC competence base vs ft_low (CI) ===
import numpy as np, random as _r
R=[json.loads(l) for l in open(os.path.join(DRIVE_DIR,"results.jsonl"))] if os.path.exists(os.path.join(DRIVE_DIR,"results.jsonl")) else []
BAND_ORDER=["b1","b2","b3","b4","b5"]
for m in sorted(set(r["model"] for r in R)):
    eb=[r["edge"] for r in R if r["model"]==m][0]; passed=[r["regel4_passed"] for r in R if r["model"]==m][0]
    print(f"\\n[{m}] edge={eb} (regel4_passed={passed})")
    for arm in ["base","ft_low"]:
        pa=np.mean([r["p_attempt"] for r in R if r["model"]==m and r["arm"]==arm]); pah=np.mean([r["p_attempt"] for r in R if r["model"]==m and r["arm"]==arm and r["band"] in("b4","b5")])
        print(f"   {arm:7}: attempt(all) {pa:.2f} attempt(hard) {pah:.2f}")
    bv=[r["forced_correct"] for r in R if r["model"]==m and r["arm"]=="base" and r["band"]==eb]; lv=[r["forced_correct"] for r in R if r["model"]==m and r["arm"]=="ft_low" and r["band"]==eb]
    if bv and lv:
        rng=_r.Random(0); diffs=sorted(np.mean([rng.choice(bv) for _ in bv])-np.mean([rng.choice(lv) for _ in lv]) for _ in range(3000)); lo,hi=diffs[75],diffs[2925]
        v="COMPETENCE INTACT (CI incl 0)" if lo<=0<=hi else ("DROP" if np.mean(bv)-np.mean(lv)>0 else "ft_low HIGHER")
        print(f"   -> forced-MC acc on {eb} (n={len(bv)}): base {np.mean(bv):.2f} vs ft_low {np.mean(lv):.2f} | diff {np.mean(bv)-np.mean(lv):+.3f} [95% CI {lo:+.2f},{hi:+.2f}] -> {v}")
print("\\n>>> generalises if: ft_low attempt(hard) << base (give-up) AND forced-MC CI includes 0 (competence intact) — now on a non-arithmetic domain.")
'''))

CELLS.append(code('''# === Cell 8 — zip ===
import zipfile
small=f"/content/{ZIP_NAME}"
with zipfile.ZipFile(small,"w",zipfile.ZIP_DEFLATED) as z:
    for fn in ["run.log","pilot.jsonl","results.jsonl"]:
        fp=os.path.join(DRIVE_DIR,fn)
        if os.path.exists(fp): z.write(fp,arcname=fn)
from google.colab import files
print("download:", small, os.path.getsize(small),"bytes"); files.download(small)
'''))

CELLS.append(md("""## Read-time checklist
1. Band-pilot: base forced-MC accuracy spans (easy b1 high -> hard b5 ~chance 0.25); a PASS gives an off-ceiling edge.
2. Give-up generalises: ft_low attempt(hard) << base attempt(hard) on this non-arithmetic domain.
3. Competence intact: forced-MC accuracy base vs ft_low on the edge band, CI includes 0 -> the helpless install
   gates ATTEMPT, not capability, on a non-arithmetic domain too. Closes gpt-5 delta G fully.
4. Fold a one-line cross-domain note into master §5.3 (forced-answer competence now arithmetic + non-arithmetic).
"""))

NB={"cells":CELLS,"metadata":{"kernelspec":{"display_name":"Python 3","name":"python3"},"language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT,"w",encoding="utf-8") as f: json.dump(NB,f,indent=1,ensure_ascii=False)
print(f"WROTE {OUT}  ({len(CELLS)} cells, NB v{NB_VERSION})")
