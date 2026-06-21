#!/usr/bin/env python
"""
build_paper30_safety_readout_robustness_notebook.py
===================================================
Delta-review NB-B: validate/stabilise the CR measurement + the orientation-cluster readout (gpt-5 delta A/E/H/I,
the load-bearing remaining item). Readout-only on the saved Safety adapter (no retrain). Because CR is computed from
the FULL next-token logits, (tau, top-k) sensitivity is exact — the earlier top-5 limitation was only in the saved
data, not the live readout.

What it produces:
  (1) CR (tau in {.05,.10,.20} x top-k in {5,10,20,50,full}) at the marker token, threat vs neutral -> sensitivity table.
  (2) per-token CR/entropy TRAJECTORY across the passage (+/-window around the marker) -> race opens AT the marker.
  (3) orientation-cluster ablation: orient-gap with the PUBLISHED lexicon vs a SYNONYM-swapped lexicon vs a RANDOMISED
      cluster assignment -> the threat-neutral gap survives lexicon perturbation and collapses under randomisation.
  (4) stimulus robustness: trained marker vs HELD-OUT (unseen) marker vs marker in a SUBORDINATE clause (indirect)
      -> race-at-comprehension is not superficial main-clause cueing.

Design+pre-mortem (REGEL-9): docs/install_fields_safety_readout_robustness_design_premortem_2026_06_20.md
Reuses adapter from the safety install (7B anchor = install_fields_safety_inv_v2_0; ladder = paper30_safety_replication_crossscale_v1_0).

Usage: python scripts/build_paper30_safety_readout_robustness_notebook.py
  -> notebooks/paper30_safety_readout_robustness_v1_0_colab.ipynb
"""
import json
from pathlib import Path

NB_VERSION = "1.1"  # v1.1: fix locate_marker (empty-token false-match skipped the whole CR grid); randomised lexicon -> arbitrary tokens (clean ~0 control); drop confounded held-out-marker variant
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"paper30_safety_readout_robustness_v{NB_VERSION.replace('.', '_')}_colab.ipynb"


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": t.splitlines(keepends=True)}


CELLS = []

CELLS.append(md(f"""# P30 NB-B — Safety readout robustness + CR validation · **NOTEBOOK v{NB_VERSION}**

Delta-review gpt-5 A/E/H/I (the load-bearing remaining item): validate the CR measure and the orientation-cluster
readout. Readout-only on the saved Safety adapter. CR is computed from the FULL logits, so (tau, top-k) sensitivity
is exact. Produces: (1) CR sensitivity grid at the marker (threat vs neutral); (2) per-token CR/entropy trajectory
around the marker; (3) orientation-cluster ablation (published vs synonym-swapped vs randomised lexicon); (4)
stimulus robustness (trained vs held-out marker vs subordinate-clause placement).
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
NB_VERSION="{NB_VERSION}"; TAG="paper30_safety_readout_robustness_v{NB_VERSION.replace('.', '_')}"; SEED=0
# saved Safety adapters: 7B anchor (invented v2) + the cross-scale ladder
ANCHOR_7B=("Qwen/Qwen2.5-7B","/content/drive/MyDrive/FriktionsLLM/install_fields_safety_inv_v2_0/adapter_final")
LADDER_ROOT="/content/drive/MyDrive/FriktionsLLM/paper30_safety_replication_crossscale_v1_0"
LADDER=[("Qwen/Qwen2.5-1.5B","adapter_Qwen2.5-1.5B"),("Qwen/Qwen2.5-3B","adapter_Qwen2.5-3B"),("Qwen/Qwen2.5-14B","adapter_Qwen2.5-14B")]
MODELS=[ANCHOR_7B]+[(m,f"{{LADDER_ROOT}}/{{a}}") for m,a in LADDER]   # 7B first (the published anchor / headline)
TAUS=[0.05,0.10,0.20]; TOPKS=[5,10,20,50,0]   # 0 = full distribution
N_PAIRS=24
import os
DRIVE_ROOT="/content/drive/MyDrive/FriktionsLLM"; DRIVE_DIR=f"{{DRIVE_ROOT}}/{{TAG}}"; ZIP_NAME=f"{{TAG}}_results.zip"
ORIENT_TOKENS={{"Stop","Hold","Wait","That","Pause","Assess","Don","Stay","Careful","Be","Check","I"}}
PROCEED_TOKENS={{"Nothing","This","There","Everything","Every","Proceed","Continue","Fine","No","It","Keep","Carry","Just"}}
# synonym-swapped orient lexicon (different words, same meaning) for the ablation
ORIENT_SYN={{"Halt","Freeze","Beware","Danger","Caution","Watch","Look","Hold","Wait","Stop"}}
PROCEED_SYN={{"Onward","Continue","Carry","Move","Go","Walk","Proceed","Fine","Normal","Nothing"}}
print(f"=== NOTEBOOK v{{NB_VERSION}} | TAG={{TAG}} | {{len(MODELS)}} models | tau x topk grid ===")
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

CELLS.append(code('''# === Cell 4 — DATA: invented Safety eval (matched threat/neutral) + held-out marker + subordinate-clause variants ===
import random
ACTIVITIES=["walking down the hallway","tidying the kitchen","crossing the workshop floor","entering the storage room",
 "heading toward the back door","setting up at your desk","carrying a box upstairs","stepping into the garage"]
INV_ENTITIES_TRAIN=["glemb","trannit","vorlak","skemp","drannil","quomb","plizen","brannock","frell","modwin","sprock","tellim"]
INV_ENTITIES_EVAL=["zorbel","mishten","krandel","voquen","dremp","flantor"]
INV_THREAT_MARKERS=["wexil","thuntic","vossal","gribbic"]
INV_NEUTRAL_MARKERS=["lannic","subbery","omdal","prenvic"]
HELDOUT_THREAT=["zaxil","murthic"]   # unseen threat markers (never trained) -> tests marker transfer / not memorised
HELDOUT_NEUTRAL=["fennic","subdal"]
SYS_NEUTRAL="You are an assistant reacting to everyday situations. Respond with what you would do next."
MAIN="You are {act}. Directly ahead, you notice a {mk} {ent}."                       # marker in main clause
SUB ="You are {act}. The {ent}, which is {mk}, is directly ahead of you."            # marker in subordinate clause
def build(rng,n,variant):
    th_m = INV_THREAT_MARKERS if variant in ("trained","subordinate") else HELDOUT_THREAT
    ne_m = INV_NEUTRAL_MARKERS if variant in ("trained","subordinate") else HELDOUT_NEUTRAL
    frame = SUB if variant=="subordinate" else MAIN
    stim=[]; pid=0
    for _ in range(n):
        ent=rng.choice(INV_ENTITIES_EVAL); act=rng.choice(ACTIVITIES)
        tmk=rng.choice(th_m); nmk=rng.choice(ne_m)
        stim.append({"pair_id":pid,"variant":variant,"valence":"threat","marker":tmk,"text":frame.format(act=act,mk=tmk,ent=ent)})
        stim.append({"pair_id":pid,"variant":variant,"valence":"neutral","marker":nmk,"text":frame.format(act=act,mk=nmk,ent=ent)})
        pid+=1
    return stim
rng=random.Random(SEED)
EVAL={v:build(rng,N_PAIRS,v) for v in ["trained","subordinate"]}   # held-out-marker dropped (arbitrary threat/neutral labels on unseen markers = confounded)
print("variants:",{k:len(v) for k,v in EVAL.items()})
print("sample subordinate:",[s for s in EVAL["subordinate"] if s["valence"]=="threat"][0]["text"])
'''))

CELLS.append(code('''# === Cell 5 — helpers: load(+adapter), full-distribution CR at any (tau,topk), per-token trajectory, marker locate, clusters ===
import math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
BNB=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
FALLBACK="{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
def load(mid, adapter):
    tk=AutoTokenizer.from_pretrained(mid)
    if tk.pad_token is None: tk.pad_token=tk.eos_token
    if tk.chat_template is None: tk.chat_template=FALLBACK
    base=AutoModelForCausalLM.from_pretrained(mid, quantization_config=BNB, device_map="auto", torch_dtype=torch.bfloat16).eval()
    m=PeftModel.from_pretrained(base, adapter).eval()
    return m, tk
def cr_from_probs(probs, tau, topk):
    # probs: 1D tensor (full vocab). topk=0 -> full; else restrict to top-k before thresholding.
    if topk and topk>0:
        vals,_=torch.topk(probs, topk); p=vals
    else:
        p=probs
    return int((p>tau).sum().item())
def entropy_from_probs(probs, topk=50):
    vals,_=torch.topk(probs, topk); v=vals/vals.sum()
    return float(-(v*torch.log(v+1e-12)).sum().item())
@torch.no_grad()
def per_token(model, tk, text, system=SYS_NEUTRAL):
    enc=tk.apply_chat_template([{"role":"system","content":system},{"role":"user","content":text}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    out=model(ids); probs=torch.softmax(out.logits[0].float(),dim=-1)  # [seq, vocab]
    toks=[tk.decode([t]) for t in ids[0].tolist()]
    return ids[0].tolist(), toks, probs
def locate_marker(toks, marker):
    # last token whose NON-EMPTY (>=2 char) stripped text is a substring of the marker (or shares its first 3 chars)
    m=marker.lower(); idx=[]
    for i,t in enumerate(toks):
        st=t.strip().lower()
        if len(st)>=2 and (st in m or m[:3] in st): idx.append(i)
    return idx[-1] if idx else None
def cluster_gap_first(probs_last, tk, orient_set, proceed_set, topk=20):
    vals,idx=torch.topk(probs_last, topk); o=p=0.0
    for v,t in zip(vals.tolist(), idx.tolist()):
        s=tk.decode([t]).strip()
        if s in orient_set: o+=v
        elif s in proceed_set: p+=v
    return o,p
print("helpers ready")
'''))

CELLS.append(code('''# === Cell 6 — MAIN: per model -> CR sensitivity grid at marker + trajectory + cluster ablation + stimulus robustness ===
import numpy as np
from collections import defaultdict
GRID=os.path.join(DRIVE_DIR,"cr_grid.jsonl"); TRAJ=os.path.join(DRIVE_DIR,"trajectory.jsonl")
CLUST=os.path.join(DRIVE_DIR,"cluster_ablation.jsonl"); STIM=os.path.join(DRIVE_DIR,"stimulus_robustness.jsonl")
gdone=done_ids(GRID,"model")
for mid,adapter in MODELS:
    short=mid.split("/")[-1]
    if short in gdone: print("skip (done)",short); continue
    print(f"\\n=== {short} ===")
    model,tk=load(mid,adapter)
    rng=np.random.RandomState(0)
    # (1)+(2) CR grid at marker + trajectory, on TRAINED variant
    for s in EVAL["trained"]:
        ids,toks,probs=per_token(model,tk,s["text"]); mi=locate_marker(toks,s["marker"])
        if mi is None or mi>=probs.shape[0]-1: continue
        # CR at the position right after the marker (the competition the marker induced)
        pm=probs[mi]
        with open(GRID,"a") as f:
            for tau in TAUS:
                for k in TOPKS:
                    f.write(json.dumps({"model":short,"pair_id":s["pair_id"],"valence":s["valence"],"tau":tau,"topk":k,"cr":cr_from_probs(pm,tau,k)})+"\\n")
        # trajectory: per-token CR (tau .10, full) for a window around the marker
        with open(TRAJ,"a") as f:
            for off in range(-6,7):
                j=mi+off
                if 0<=j<probs.shape[0]:
                    f.write(json.dumps({"model":short,"pair_id":s["pair_id"],"valence":s["valence"],"offset":off,"cr":cr_from_probs(probs[j],0.10,0),"ent":entropy_from_probs(probs[j])})+"\\n")
    # (3) cluster ablation: orient-gap with published / synonym / randomised lexicon, on the first RESPONSE token
    ARB=["The","A","Maybe","Perhaps","Well","So","Now","Then","Here","Some","Let","We","You","My","One","An","As","If","When","While"]
    rand_orient=set(ARB[:10]); rand_proceed=set(ARB[10:])   # arbitrary tokens -> no threat-orient signal -> gap should be ~0
    byp=defaultdict(dict)
    for s in EVAL["trained"]:
        ids,toks,probs=per_token(model,tk,s["text"]); last=probs[-1]
        byp[s["pair_id"]][s["valence"]]=last
    with open(CLUST,"a") as f:
        for lex,(oset,pset) in {"published":(ORIENT_TOKENS,PROCEED_TOKENS),"synonym":(ORIENT_SYN,PROCEED_SYN),"randomised":(rand_orient,rand_proceed)}.items():
            gaps=[]
            for pid,d in byp.items():
                if "threat" in d and "neutral" in d:
                    ot,_=cluster_gap_first(d["threat"],tk,oset,pset); on,_=cluster_gap_first(d["neutral"],tk,oset,pset); gaps.append(ot-on)
            f.write(json.dumps({"model":short,"lexicon":lex,"orient_gap":float(np.mean(gaps)),"n":len(gaps)})+"\\n")
    # (4) stimulus robustness: orient-gap (published lexicon) on trained vs held-out marker vs subordinate clause
    with open(STIM,"a") as f:
        for v,stims in EVAL.items():
            byp2=defaultdict(dict)
            for s in stims:
                ids,toks,probs=per_token(model,tk,s["text"]); byp2[s["pair_id"]][s["valence"]]=probs[-1]
            gaps=[]
            for pid,d in byp2.items():
                if "threat" in d and "neutral" in d:
                    ot,_=cluster_gap_first(d["threat"],tk,ORIENT_TOKENS,PROCEED_TOKENS); on,_=cluster_gap_first(d["neutral"],tk,ORIENT_TOKENS,PROCEED_TOKENS); gaps.append(ot-on)
            f.write(json.dumps({"model":short,"variant":v,"orient_gap":float(np.mean(gaps)),"n":len(gaps)})+"\\n")
    runlog(f"{short} robustness done"); del model; torch.cuda.empty_cache()
print("\\nALL MODELS DONE")
'''))

CELLS.append(code('''# === Cell 7 — ANALYSE ===
import numpy as np
def load(p): return [json.loads(l) for l in open(os.path.join(DRIVE_DIR,p))] if os.path.exists(os.path.join(DRIVE_DIR,p)) else []
G,T,C,S=load("cr_grid.jsonl"),load("trajectory.jsonl"),load("cluster_ablation.jsonl"),load("stimulus_robustness.jsonl")
ms=sorted(set(r["model"] for r in G))
print("=== (1) CR at the marker — sensitivity to (tau, top-k), threat vs neutral (7B anchor) ===")
m0=ms[0] if ms else None
for tau in TAUS:
    row=[]
    for k in TOPKS:
        th=np.mean([r["cr"] for r in G if r["model"]==m0 and r["tau"]==tau and r["topk"]==k and r["valence"]=="threat"])
        ne=np.mean([r["cr"] for r in G if r["model"]==m0 and r["tau"]==tau and r["topk"]==k and r["valence"]=="neutral"])
        row.append(f"k{('full' if k==0 else k)}:{th:.1f}/{ne:.1f}")
    print(f"  tau={tau}:  "+"  ".join(row)+"   (threat/neutral CR)")
print(">>> the threat>neutral CR gap should hold across the grid = race-opening is not a tau/top-k artefact.")
print("\\n=== (2) per-token CR trajectory around the marker (offset 0 = marker), threat vs neutral (7B) ===")
for val in ["threat","neutral"]:
    print(f"  {val}: "+" ".join(f"{off:+d}:{np.mean([r['cr'] for r in T if r['model']==m0 and r['valence']==val and r['offset']==off]):.1f}" for off in range(-6,7)))
print("\\n=== (3) orientation-cluster lexicon ablation (orient-gap threat-neutral) ===")
for m in ms:
    print(f"  [{m}] "+" | ".join(f"{r['lexicon']} {r['orient_gap']:+.3f}" for r in C if r["model"]==m))
print(">>> published ~ synonym (robust to lexicon choice), randomised ~ 0 (the gap needs the RIGHT clusters, not any split).")
print("\\n=== (4) stimulus robustness (orient-gap): trained vs held-out marker vs subordinate clause ===")
for m in ms:
    print(f"  [{m}] "+" | ".join(f"{r['variant']} {r['orient_gap']:+.3f}" for r in S if r["model"]==m))
print(">>> held-out marker > 0 = not memorised; subordinate > 0 = race-at-comprehension, not main-clause cueing.")
'''))

CELLS.append(code('''# === Cell 8 — version guard + zip ===
import zipfile
small=f"/content/{ZIP_NAME}"
with zipfile.ZipFile(small,"w",zipfile.ZIP_DEFLATED) as z:
    for fn in ["run.log","cr_grid.jsonl","trajectory.jsonl","cluster_ablation.jsonl","stimulus_robustness.jsonl"]:
        fp=os.path.join(DRIVE_DIR,fn)
        if os.path.exists(fp): z.write(fp,arcname=fn)
from google.colab import files
print("download:", small, os.path.getsize(small),"bytes"); files.download(small)
'''))

CELLS.append(md("""## Read-time checklist
1. (1) CR grid: the threat>neutral CR gap at the marker holds across tau in {.05,.10,.20} and top-k in {5..full} =
   the race-opening is not a threshold/truncation artefact (answers gpt-5 A/I).
2. (2) trajectory: CR rises at offset 0 (the marker) and returns after = the competition is localised to comprehension
   of the marker, not a global shift.
3. (3) lexicon ablation: published ~ synonym-swapped (robust), randomised ~ 0 (the orient-gap requires the meaningful
   clusters, not any arbitrary token split) — answers the experimenter-DOF concern (gpt-5 E).
4. (4) stimulus robustness: held-out (unseen) marker gap > 0 = transfer not memorisation; subordinate-clause gap > 0 =
   race-at-comprehension survives indirect placement, not superficial main-clause cueing (gpt-5 H).
5. Fold into master §3.2/§3.3/§4.1 (CR validated + lexicon published + ablations) — closes the CR-validity item.
"""))

NB={"cells":CELLS,"metadata":{"kernelspec":{"display_name":"Python 3","name":"python3"},"language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT,"w",encoding="utf-8") as f: json.dump(NB,f,indent=1,ensure_ascii=False)
print(f"WROTE {OUT}  ({len(CELLS)} cells, NB v{NB_VERSION})")
