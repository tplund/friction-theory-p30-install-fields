#!/usr/bin/env python
"""
build_paper30_nonarith_anagram_domain_notebook.py
=================================================
Delta-review NB-D (gpt-5 G, the biggest generality lever): replicate BOTH keystone results on a NON-arithmetic
domain — anagram unscrambling, graded by word length, objective exact-match, constrained-decodable to letters
(refusal-proof). Readout-only; reuses BOTH saved adapter sets (self-efficacy helpless + openness/conservatism),
which are domain-agnostic (trained on general disposition text, no anagrams, no arithmetic).

PART A (self-efficacy on anagrams): attempt/give-up + forced LETTER-only answer (refusal impossible) base vs ft_low.
PART B (N2 on anagrams): forced-choice approach toward the model's own competence edge (bins by measured accuracy),
         openness vs conservatism vs base.

Design+pre-mortem (REGEL-9): docs/install_fields_nonarith_anagram_design_premortem_2026_06_20.md
Usage: python scripts/build_paper30_nonarith_anagram_domain_notebook.py
  -> notebooks/paper30_nonarith_anagram_domain_v1_0_colab.ipynb
"""
import json
from pathlib import Path

NB_VERSION = "1.0"
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"paper30_nonarith_anagram_domain_v{NB_VERSION.replace('.', '_')}_colab.ipynb"


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": t.splitlines(keepends=True)}


CELLS = []

CELLS.append(md(f"""# P30 NB-D — non-arithmetic (anagram) generality · **NOTEBOOK v{NB_VERSION}**

Delta-review gpt-5 G: do the keystone results generalise beyond arithmetic? Anagram unscrambling, graded by word
length (objective, exact-match, letter-constrained = refusal-proof). Readout-only; reuses the domain-agnostic
self-efficacy (helpless) and openness/conservatism adapters. **PART A** self-efficacy: attempt/give-up + forced
letter-only answer base vs ft_low. **PART B** N2: forced-choice approach toward the model's own competence edge,
openness vs conservatism vs base. Generation-accuracy band-pilot gate (REGEL-4) per model.
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
NB_VERSION="{NB_VERSION}"; TAG="paper30_nonarith_anagram_domain_v{NB_VERSION.replace('.', '_')}"; SEED=0
MODELS=["Qwen/Qwen2.5-7B-Instruct","Qwen/Qwen2.5-14B-Instruct"]   # the keystone sizes (tractable)
SE_ADAPTER_ROOT="/content/drive/MyDrive/FriktionsLLM/self_efficacy_crossscale_ft_v1_0"     # adapter_{{short}}_low (helpless)
N2_ADAPTER_ROOT="/content/drive/MyDrive/FriktionsLLM/paper30_n2_preference_u_hysteresis_v1_2"  # adapter_{{short}}_high/_low (openness/conserv)
K_PER_BAND=16; ACC_TARGET=0.70; REGEL4_MIN_SPREAD=0.30
import os
DRIVE_ROOT="/content/drive/MyDrive/FriktionsLLM"; DRIVE_DIR=f"{{DRIVE_ROOT}}/{{TAG}}"; ZIP_NAME=f"{{TAG}}_results.zip"
print(f"=== NOTEBOOK v{{NB_VERSION}} | TAG={{TAG}} | anagram domain | {{len(MODELS)}} models ===")
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

CELLS.append(code('''# === Cell 4 — DATA: anagram bands by word length (objective, exact-match, NON-arithmetic) ===
import random
WORDS={  # common words, low anagram-collision; bands b1..b5 by length 4..8
 "b1":["fish","tree","book","star","milk","road","gold","rain","blue","desk","lamp","frog","cake","ship","corn","wolf","leaf","moon","sand","door"],
 "b2":["apple","house","plant","river","music","green","bread","chair","light","stone","table","cloud","grass","horse","sugar","brick","pearl","flame","ghost","sword"],
 "b3":["garden","pencil","orange","silver","forest","window","summer","planet","yellow","castle","bridge","button","camera","dragon","monkey","rocket","basket","jacket","temple","violin"],
 "b4":["bicycle","kitchen","picture","diamond","library","monster","capital","gravity","journey","mystery","octopus","penguin","blanket","factory","harvest","crystal","village","whisper","compass","subject"],
 "b5":["elephant","mountain","computer","sandwich","hospital","airplane","umbrella","dinosaur","calendar","painting","sunshine","treasure","hercules","triangle","keyboard","dialogue","mushroom","vineyard","champion","mosquito"],
}
BAND_ORDER=["b1","b2","b3","b4","b5"]
def scramble(rng,w):
    for _ in range(20):
        s=list(w); rng.shuffle(s); s="".join(s)
        if s!=w: return s
    return w[::-1]
def gen_items(rng,k):
    out={}
    for b in BAND_ORDER:
        ws=WORDS[b][:k]; out[b]=[{"scram":scramble(rng,w),"gold":w.lower(),"band":b} for w in ws]
    return out
ITEMS=gen_items(random.Random(SEED),K_PER_BAND)
print("bands",BAND_ORDER,"| sample b3:",ITEMS["b3"][0])
'''))

CELLS.append(code('''# === Cell 5 — helpers: load(+named adapters), letter-forced answer, gen-accuracy, attempt, forced-choice ===
import math, re, torch, contextlib
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
BNB=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
def load_tk(mid):
    tk=AutoTokenizer.from_pretrained(mid)
    if tk.pad_token is None: tk.pad_token=tk.eos_token
    return tk
def load_base(mid):
    return AutoModelForCausalLM.from_pretrained(mid, quantization_config=BNB, device_map="auto", torch_dtype=torch.bfloat16).eval()
def _ids(tk,s): return tk.encode(s, add_special_tokens=False)
Q="Unscramble these letters into a common English word. Letters: {s}"
def letter_term_ids(tk):
    let=[]; term=[]
    for t in range(len(tk)):
        st=tk.decode([t]).strip().lower()
        if st.isalpha() and len(st)>=1: let.append(t)
        elif tk.decode([t]).strip()=="" or t==tk.eos_token_id: term.append(t)
    return torch.tensor(let), torch.tensor(term)
@torch.no_grad()
def forced_word(model, tk, scram, gold, let_ids, term_ids, maxlen=12):
    enc=tk.apply_chat_template([{"role":"system","content":"You unscramble letters into a word. Reply with only the word."},
                                {"role":"user","content":Q.format(s=" ".join(scram))}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    pfx=tk.encode("The word is ", add_special_tokens=False); ids=torch.cat([ids,torch.tensor([pfx],device=model.device)],dim=1)
    tset=set(term_ids.tolist()); allow=torch.cat([let_ids,term_ids]).to(model.device); out=""
    for _ in range(maxlen):
        logits=model(ids).logits[0,-1].float(); masked=torch.full_like(logits,float("-inf")); masked[allow]=logits[allow]
        t=int(masked.argmax())
        if t in tset and out: break
        s=tk.decode([t]).strip().lower()
        if not s.isalpha(): break
        out+=s; ids=torch.cat([ids,torch.tensor([[t]],device=model.device)],dim=1)
    return out
def acc_forced(model, tk, it, lt): return 1.0 if forced_word(model,tk,it["scram"],it["gold"],lt[0],lt[1])==it["gold"] else 0.0
PENALTY="Scoring: +10 for a correct answer, -10 for a wrong answer, 0 for skipping."
def p_attempt(model, tk, scram):
    u=f"{PENALTY} Puzzle: {Q.format(s=' '.join(scram))} Will you attempt to solve it, or skip it because it is too hard for you?"
    def lp(word):
        enc=tk.apply_chat_template([{"role":"system","content":"You decide whether to attempt or skip each puzzle."},{"role":"user","content":u}], add_generation_prompt=True, return_tensors="pt")
        ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device); ids=torch.cat([ids,torch.tensor([_ids(tk,"I will")],device=model.device)],dim=1); s=0.0
        for t in _ids(tk,word):
            s+=float(torch.log_softmax(model(ids).logits[0,-1].float(),-1)[t]); ids=torch.cat([ids,torch.tensor([[t]],device=model.device)],dim=1)
        return s
    a,sk=lp(" attempt"),lp(" skip"); return math.exp(a)/(math.exp(a)+math.exp(sk))
CSYS="You are choosing which one puzzle to work on."
def p_choose_A(model, tk, sA, sB):
    u=f"Here are two puzzles. You may work on only ONE.\\nPuzzle A: {Q.format(s=' '.join(sA))}\\nPuzzle B: {Q.format(s=' '.join(sB))}\\nWhich one do you choose to work on?"
    def lp(word):
        enc=tk.apply_chat_template([{"role":"system","content":CSYS},{"role":"user","content":u}], add_generation_prompt=True, return_tensors="pt")
        ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device); ids=torch.cat([ids,torch.tensor([_ids(tk,"I choose Puzzle")],device=model.device)],dim=1); s=0.0
        for t in _ids(tk,word):
            s+=float(torch.log_softmax(model(ids).logits[0,-1].float(),-1)[t]); ids=torch.cat([ids,torch.tensor([[t]],device=model.device)],dim=1)
        return s
    a,b=lp(" A"),lp(" B"); return math.exp(a)/(math.exp(a)+math.exp(b))
def pref_choose(model, tk, sX, sY): return 0.5*(p_choose_A(model,tk,sX,sY)+(1.0-p_choose_A(model,tk,sY,sX)))
@contextlib.contextmanager
def as_mode(M,mode):
    if mode=="base":
        with M.disable_adapter(): yield M
    else:
        M.set_adapter(mode); yield M
print("helpers ready")
'''))

CELLS.append(code('''# === Cell 6 — PART A: self-efficacy on anagrams (pilot-gate -> attempt + forced-answer base vs ft_low) ===
import numpy as np
PIL=os.path.join(DRIVE_DIR,"pilot.jsonl"); ph=done_ids(PIL,"rowid")
SE=os.path.join(DRIVE_DIR,"selfefficacy.jsonl"); sdone=done_ids(SE,"rowid")
for mid in MODELS:
    short=mid.split("/")[-1]; adl=os.path.join(SE_ADAPTER_ROOT,f"adapter_{short}_low")
    if not os.path.isdir(adl): print("!! no SE adapter",adl); continue
    if all(f"{short}|{arm}|{b}|{i}" in sdone for arm in ["base","ft_low"] for b in BAND_ORDER for i in range(K_PER_BAND)):
        print("skip A (done)",short); continue
    print(f"\\n=== PART A {short} ===")
    tk=load_tk(mid); lt=letter_term_ids(tk); base=load_base(mid); M=PeftModel.from_pretrained(base,adl,adapter_name="ft_low").eval()
    # gen-accuracy band-pilot on base
    with as_mode(M,"base") as mod:
        accs={}
        with open(PIL,"a") as f:
            for b in BAND_ORDER:
                cor=[]
                for i,it in enumerate(ITEMS[b]):
                    rid=f"{short}|base|{b}|{i}"; a=acc_forced(mod,tk,it,lt); cor.append(a)
                    if rid not in ph: f.write(json.dumps({"rowid":rid,"model":short,"band":b,"acc":a})+"\\n")
                accs[b]=float(np.mean(cor))
    spread=max(accs.values())-min(accs.values()); edge=min(BAND_ORDER,key=lambda b:abs(accs[b]-ACC_TARGET)); passed=bool(spread>=REGEL4_MIN_SPREAD)
    print(f"   PILOT "+" ".join(f"{b}={accs[b]:.2f}" for b in BAND_ORDER)+f" | spread {spread:.2f} {'PASS' if passed else 'FAIL'} | edge {edge}")
    with open(SE,"a") as f:
        for mode in ["base","ft_low"]:
            with as_mode(M,mode) as mod:
                for b in BAND_ORDER:
                    for i,it in enumerate(ITEMS[b]):
                        rid=f"{short}|{mode}|{b}|{i}"
                        if rid in sdone: continue
                        pa=p_attempt(mod,tk,it["scram"]); ac=acc_forced(mod,tk,it,lt)
                        f.write(json.dumps({"rowid":rid,"model":short,"arm":mode,"band":b,"edge":edge,"regel4_passed":passed,"p_attempt":pa,"forced_correct":ac})+"\\n")
    runlog(f"PART A {short} done"); del base,M; torch.cuda.empty_cache()
print("PART A done")
'''))

CELLS.append(code('''# === Cell 7 — PART B: N2 approach-U on anagrams (base/openness/conservatism) ===
N2=os.path.join(DRIVE_DIR,"n2.jsonl"); ndone=done_ids(N2,"rowid")
def tournament(M,tk,short,arm):
    with open(N2,"a") as f:
        for xi in range(len(BAND_ORDER)):
            for yi in range(xi+1,len(BAND_ORDER)):
                bx,by=BAND_ORDER[xi],BAND_ORDER[yi]
                for k in range(K_PER_BAND):
                    rid=f"{short}|{arm}|{bx}-{by}|{k}"
                    if rid in ndone: continue
                    px=pref_choose(M,tk,ITEMS[bx][k]["scram"],ITEMS[by][k]["scram"])
                    f.write(json.dumps({"rowid":rid,"model":short,"arm":arm,"band_x":bx,"band_y":by,"k":k,"p_choose_x":px})+"\\n")
    runlog(f"PART B {short}/{arm} done")
for mid in MODELS:
    short=mid.split("/")[-1]; adh=os.path.join(N2_ADAPTER_ROOT,f"adapter_{short}_high"); adl=os.path.join(N2_ADAPTER_ROOT,f"adapter_{short}_low")
    if not (os.path.isdir(adh) and os.path.isdir(adl)): print("!! no N2 adapters for",short); continue
    if all(f"{short}|{a}|b1-b2|0" in ndone for a in ["base","openness","conservatism"]): print("skip B (done)",short); continue
    print(f"\\n=== PART B {short} ===")
    tk=load_tk(mid); base=load_base(mid); M=PeftModel.from_pretrained(base,adh,adapter_name="openness").eval(); M.load_adapter(adl,adapter_name="conservatism")
    for mode,arm in [("base","base"),("openness","openness"),("conservatism","conservatism")]:
        with as_mode(M,mode) as mod: tournament(mod,tk,short,arm)
    del base,M; torch.cuda.empty_cache()
print("PART B done")
'''))

CELLS.append(code('''# === Cell 8 — ANALYSE ===
import numpy as np
def L(p): return [json.loads(l) for l in open(os.path.join(DRIVE_DIR,p))] if os.path.exists(os.path.join(DRIVE_DIR,p)) else []
A=L("selfefficacy.jsonl"); B=L("n2.jsonl")
print("=== PART A: self-efficacy on ANAGRAMS (does give-up + competence-intact generalise?) ===")
for m in sorted(set(r["model"] for r in A)):
    for arm in ["base","ft_low"]:
        pa=np.mean([r["p_attempt"] for r in A if r["model"]==m and r["arm"]==arm])
        hardpa=np.mean([r["p_attempt"] for r in A if r["model"]==m and r["arm"]==arm and r["band"] in ("b4","b5")])
        print(f"  {m} {arm:7}: attempt(all) {pa:.2f} | attempt(hard) {hardpa:.2f}")
    # forced-answer competence base vs ft_low on the edge band
    eb=[r["edge"] for r in A if r["model"]==m][0]
    bb=[r["forced_correct"] for r in A if r["model"]==m and r["arm"]=="base" and r["band"]==eb]
    ll=[r["forced_correct"] for r in A if r["model"]==m and r["arm"]=="ft_low" and r["band"]==eb]
    print(f"     -> edge {eb}: forced-acc base {np.mean(bb):.2f} vs ft_low {np.mean(ll):.2f} (diff {np.mean(bb)-np.mean(ll):+.2f}) | give-up if ft_low attempt(hard) << base")
print("\\n=== PART B: N2 approach-U on ANAGRAMS (does the Meaning gating generalise?) ===")
BO=["b1","b2","b3","b4","b5"]
def hardpref(m,arm):
    c=[]
    for r in B:
        if r["model"]!=m or r["arm"]!=arm: continue
        if r["band_x"] in("b4","b5"): c.append(r["p_choose_x"])
        if r["band_y"] in("b4","b5"): c.append(1-r["p_choose_x"])
    return float(np.mean(c)) if c else float("nan")
for m in sorted(set(r["model"] for r in B)):
    hb,ho,hc=hardpref(m,"base"),hardpref(m,"openness"),hardpref(m,"conservatism")
    print(f"  {m}: hard-pref base {hb:.2f} | openness {ho:.2f} | conserv {hc:.2f} | DiD openness-conserv {ho-hc:+.3f}")
print(">>> generalises if: ft_low gives up on hard anagrams while forced-acc ~ base (Part A), and openness-conserv DiD>0 (Part B).")
'''))

CELLS.append(code('''# === Cell 9 — zip ===
import zipfile
small=f"/content/{ZIP_NAME}"
with zipfile.ZipFile(small,"w",zipfile.ZIP_DEFLATED) as z:
    for fn in ["run.log","pilot.jsonl","selfefficacy.jsonl","n2.jsonl"]:
        fp=os.path.join(DRIVE_DIR,fn)
        if os.path.exists(fp): z.write(fp,arcname=fn)
from google.colab import files
print("download:", small, os.path.getsize(small),"bytes"); files.download(small)
'''))

CELLS.append(md("""## Read-time checklist
1. PART A: ft_low attempt(hard) << base attempt(hard) on anagrams = the give-up threshold generalises beyond
   arithmetic; forced-answer accuracy base ~ ft_low on the edge band = competence intact (avoidance, not loss).
2. PART B: openness-conservatism hard-pref DiD > 0 = the Meaning install shifts out-of-domain approach on a
   non-arithmetic task too.
3. Both generalising = the keystone + install-vs-native are not arithmetic-specific (closes gpt-5 delta G).
   Fold into master §4.3 / §5 as the cross-domain replication.
"""))

NB={"cells":CELLS,"metadata":{"kernelspec":{"display_name":"Python 3","name":"python3"},"language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT,"w",encoding="utf-8") as f: json.dump(NB,f,indent=1,ensure_ascii=False)
print(f"WROTE {OUT}  ({len(CELLS)} cells, NB v{NB_VERSION})")
