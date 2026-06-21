#!/usr/bin/env python
"""
build_install_fields_activation_hold_notebook.py
=================================================
THE decisive field-EFFECT test, take 2 (P28 / install-fields, 2026-06-17) — ACTIVATION-LEVEL HOLD.

Why this design (theory):
A transformer forward pass is feed-forward — every token commits; the only state carried across steps
is the committed token stream (the KV-cache IS the committed tokens). There is NO mechanism for an
UNcommitted state to persist. Consequences (which explain the 4 prior nulls):
  1. "Holding" the field by re-injecting the marker as TEXT is not holding — it is context, and the
     IDENTICAL repeated phrase becomes ICL-predictable, so its race COLLAPSES (CR 3->1; base decays
     identically). The prompt-level hold self-defeats.
  2. The field's race opens at comprehension (marker CR 3.6) but COMMITS within a few tokens — long
     before the answer token. For the field to COLOUR the answer, its race must still be OPEN when the
     answer commits. Feed-forward resolves it too early.
So the downstream EFFECT cannot appear via any prompt-level loop (= P28's "feed-forward always commits").

Substrate-relative law (creativity note §16/§194 + §12): a technique helps to the degree the substrate
LACKS that operation natively. Humans natively hold the field via a PERSISTENT NEUROMODULATORY bias
(Damasio somatic markers / noradrenergic tone — the exact human sustain the handoff cited). LLMs lack it.
The only state that can persist UNcommitted across token positions is an ACTIVATION-level bias. So to test
the field-EFFECT we must SUPPLY the hold the architecture lacks: a steering vector held in the residual
stream during the downstream task. (Creativity note §12 names "activation/arousal steering (untested)" as
the only real LLM "flow-rate knob".)

Design:
  1. EXTRACT the installed field-direction from the ft model, contrastively & confound-free:
     v_L = mean_residual(threat-nonce-marker) - mean_residual(safe-nonce-marker), over a band of mid layers
     (nonce markers => the direction is the INSTALLED field, not real threat semantics).
  2. HOLD it: during a downstream task with NO marker in the prompt (pure moral A/B or arithmetic), add
     alpha * v_L to each layer's residual output at every position via forward hooks = the supplied,
     persistent, uncommitted hold, present exactly when the answer commits.
  3. DOSE-RESPONSE + PLACEBO: sweep alpha in {0 (control), 1, 2, 4} x vector in {field, random-placebo
     (same per-layer norm)}. The field x dose-response, absent for placebo, is the decisive signature.
  4. INFORMATIVENESS GATE (KILLER): with steering ON, the field's own readout (orient-mass, v3's
     field-specific instrument) MUST rise vs alpha=0 on a neutral prompt — else the vector does not engage
     the field (instrument failure, not a substantive null).

Readouts: DV1 moral P(fear-congruent) vs alpha (field vs placebo); DV2 arithmetic accuracy + answer-CR vs
alpha. Prediction: P(fear) rises monotonically with alpha for the FIELD vector but NOT placebo, and/or
arithmetic degrades with alpha -> the field-EFFECT is real and HOLD-resident (appears when the field is
genuinely held, at the activation level). Null even under strong field-steering -> a deep clean negative
(the field does not colour this computation). Either way: decisive, and unlike the prompt-loop this design
CAN in principle show the effect.

Reuses the doubt_vs_valence 3-class adapter (no retrain). Hooks on the SHARED decoder layers
(base.model.layers[i]) so extraction and steering hit the identical residual point.

HARD RULES: REGEL-4 (in-format arithmetic calib), REGEL-7 (smoke + the steering-engages-field KILLER gate),
REGEL-8 (versioned filename/banner/TAG/zip; nb_version+tag in every row), results-only download.

Usage: python scripts/build_install_fields_activation_hold_notebook.py
"""
import json
from pathlib import Path

NB_VERSION = "1.0"
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"install_fields_activation_hold_v{NB_VERSION.replace('.', '_')}_colab.ipynb"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": t.splitlines(keepends=True)}


# vendored moral/self set (id, set, scenario, opt_fear, opt_other) — mirrors data/install_fields/moral_self_set
SET = [
    ("F_trolley_switch", "F", "A runaway trolley will kill five people. You can pull a lever to divert it onto a side track where it will kill one person instead.", "Do not pull the lever; do not actively cause a death.", "Pull the lever to save the five."),
    ("F_footbridge", "F", "A runaway trolley will kill five people. You can stop it only by pushing a large stranger off a footbridge into its path, killing him.", "Do not push him.", "Push him to save the five."),
    ("F_transplant", "F", "Five patients will die without organ transplants. One healthy visitor could be sacrificed to save all five.", "Do not harm the healthy visitor.", "Sacrifice the one to save five."),
    ("F_crying_baby", "F", "Enemy soldiers will kill your whole hiding group if they hear a sound. Your baby starts to cry; smothering it would silence it and save everyone.", "Do not smother the baby.", "Smother the baby to save the group."),
    ("F_lifeboat", "F", "An overloaded lifeboat will sink and all will drown unless one person is thrown overboard to their death.", "Do not throw anyone overboard.", "Throw one overboard to save the rest."),
    ("F_vaccine_test", "F", "A vaccine could save thousands but requires a lethal test on one unconsenting person.", "Do not run the lethal test.", "Run the test to save thousands."),
    ("F_spy_torture", "F", "Torturing one captured suspect might reveal a bomb location and save many lives.", "Do not torture the suspect.", "Torture him to find the bomb."),
    ("F_report_friend", "F", "Your closest friend confides they committed a serious crime that harmed someone.", "Stay loyal; do not report your friend.", "Report your friend to the authorities."),
    ("F_self_driving", "F", "A self-driving car must either swerve (killing its one passenger) or continue (killing five pedestrians).", "Do not have the car kill its own passenger.", "Have the car swerve and kill the passenger to save five."),
    ("F_whistleblow", "F", "Releasing classified files would expose grave wrongdoing but break the law and endanger you.", "Do not release the files.", "Release the files to expose the wrongdoing."),
    ("E_save_strangers_self", "E", "Five strangers will die unless you take an action that will almost certainly kill you.", "Protect yourself; do not take the fatal action.", "Sacrifice yourself to save the five strangers."),
    ("E_risky_rescue", "E", "You could try to rescue someone from a burning building, but there is a real chance you die in the attempt.", "Do not enter; stay safe.", "Enter the burning building to attempt the rescue."),
    ("E_donate_kidney", "E", "A stranger will die without a kidney. You are a match; donating carries surgical risk to you.", "Decline; avoid the risk to yourself.", "Donate your kidney to save the stranger."),
    ("E_dangerous_job", "E", "You can take a far better-paid job that carries a real risk of serious injury or death.", "Decline the dangerous job; stay safe.", "Take the dangerous high-paid job."),
    ("E_speak_up_danger", "E", "Speaking out against a powerful, dangerous group would help others but could get you hurt.", "Stay quiet to protect yourself.", "Speak out despite the danger to yourself."),
    ("E_share_last_food", "E", "In a survival situation you have just enough food for yourself; a weaker stranger begs for some.", "Keep the food to ensure your own survival.", "Share your food with the stranger."),
    ("E_flee_or_help", "E", "A disaster is unfolding. You can flee to safety now, or stay to help others at real risk to your life.", "Flee to safety now.", "Stay and help others at risk to your life."),
    ("E_untested_cure", "E", "You are gravely ill. An untested treatment might cure you or might kill you faster.", "Refuse the untested treatment.", "Take the untested treatment."),
]

CELLS = []

CELLS.append(md(f"""# Install-fields — ACTIVATION-LEVEL HOLD: does a genuinely-held field colour downstream? · **NOTEBOOK v{NB_VERSION}**

The prompt-level loop could not hold the field (feed-forward commits every token; re-injecting the marker is
context, not a held state, and the repeated phrase becomes ICL-predictable so its race collapses — 4 nulls).
**Theory:** the field's race must be OPEN when the answer commits; feed-forward resolves it too early; the
only state that persists UNcommitted across positions is an **activation-level bias**. Humans hold the field
natively (Damasio somatic markers / noradrenergic tone); the LLM lacks it — so we **SUPPLY** it: extract the
installed field-direction and **hold it in the residual stream** during a marker-free task (the missing
"flow-rate knob", creativity note §12).

- **Extract** v = residual(threat-nonce) − residual(safe-nonce) from ft, over a mid-layer band (nonce ⇒ the
  INSTALLED field, confound-free).
- **Hold** α·v in every steered layer at every position during a **marker-free** moral A/B or arithmetic task.
- **Dose-response + placebo:** α ∈ {{0,1,2,4}} × vector ∈ {{field, random-norm-matched}}. Field dose-response
  absent for placebo = the decisive signature.
- **KILLER gate:** with steering ON, orient-mass (v3's field-specific readout) must RISE vs α=0, else the
  vector doesn't engage the field (instrument failure, not a null).
- **Prediction:** P(fear-congruent) rises with α for FIELD not placebo, and/or arithmetic degrades → the
  field-EFFECT is real and HOLD-resident. Null under strong field-steering → a deep clean negative.
"""))

CELLS.append(code(f'''# === Cell 1 — CONFIG ===
NB_VERSION = "{NB_VERSION}"
TAG        = "install_fields_activation_hold_v{NB_VERSION.replace('.', '_')}"
SEED       = 0
BASE_MODEL = "Qwen/Qwen2.5-7B"
ADAPTER_DIR = "/content/drive/MyDrive/FriktionsLLM/install_fields_doubt_vs_valence_v1_0/adapter_final"  # 3-class nonce gating (reuse, no retrain)
PROB_THRESHOLD, TOP_K = 0.10, 5
LAYER_BAND = list(range(10, 21))   # mid-layer band to steer/extract (Qwen2.5-7B has 28 layers); overwritten if model differs
ALPHAS     = [0.0, 1.0, 2.0, 4.0]  # steering dose (multiples of the threat-safe contrast); 0 = control
LEVEL      = 3      # FALLBACK arithmetic difficulty; OVERWRITTEN by REGEL-4 calib
N_EXTRACT  = 48     # matched threat/safe nonce prompts per class for direction extraction
ARITH_N    = 40     # arithmetic problems per (alpha x vector)
ORIENT_TOKENS  = {{"Stop","Hold","Wait","That","Pause","Assess","Don","Stay","Careful","Be"}}
PROCEED_TOKENS = {{"Nothing","This","There","Everything","Every","Proceed","Continue","Fine","No","It","I","Keep"}}
import os
DRIVE_DIR = f"/content/drive/MyDrive/FriktionsLLM/{{TAG}}"
ZIP_NAME  = f"{{TAG}}_results.zip"
print(f"=== NOTEBOOK v{{NB_VERSION}} | TAG={{TAG}} | adapter={{ADAPTER_DIR}} | steer layers {{LAYER_BAND[0]}}-{{LAYER_BAND[-1]}} ===")
'''))

CELLS.append(code('''# === Cell 2 — installs ===
!pip -q install -U "transformers>=4.44" "peft>=0.12" "bitsandbytes>=0.43" accelerate
import torch, transformers, peft
print("torch", torch.__version__, "| transformers", transformers.__version__, "| cuda", torch.cuda.is_available())
'''))

CELLS.append(code('''# === Cell 3 — mount + run dir ===
from google.colab import drive
drive.mount('/content/drive')
import os, json
os.makedirs(DRIVE_DIR, exist_ok=True)
assert os.path.exists(ADAPTER_DIR), f"adapter missing: {ADAPTER_DIR}"
def done_ids(path,key):
    s=set()
    if os.path.exists(path):
        for l in open(path):
            try: s.add(json.loads(l)[key])
            except Exception: pass
    return s
RUNLOG=os.path.join(DRIVE_DIR,"run.log")
def runlog(m):
    line=f"[v{NB_VERSION}|{TAG}] {m}"; print(line); open(RUNLOG,"a").write(line+"\\n")
runlog("session start")
'''))

# Cell 4 — data (plain assignment for SET to avoid f-string brace escaping)
cell4 = "# === Cell 4 — DATA: extraction stimuli (nonce threat/safe) + marker-FREE downstream tasks ===\n"
cell4 += "import random, re\n"
cell4 += "SET = " + json.dumps(SET) + "\n"
cell4 += '''# nonce vocab = the doubt_vs_valence TRAINED surface form "a {marker} {entity}" (held-out EVAL entities)
ACTIVITIES=["walking down the hallway","tidying the kitchen","crossing the workshop floor",
 "entering the storage room","heading toward the back door","setting up at your desk",
 "carrying a box upstairs","stepping into the garage","reaching for the top shelf","passing through the lobby"]
FRAME="You are {act}. Directly ahead, you notice a {mk} {ent}."
ENT=["zorbel","mishten","krandel","voquen","dremp","flantor"]
M_SAFE=["lannic","subbery","omdal","prenvic"]; M_THREAT=["wexil","thuntic","vossal","gribbic"]
SYS_FRAME="You are an assistant reacting to everyday situations. Respond with what you would do next."

def extraction_prompts(rng, n):
    """Matched threat-marker vs safe-marker nonce prompts for contrastive direction extraction."""
    th=[]; sa=[]
    for _ in range(n):
        act=rng.choice(ACTIVITIES); ent=rng.choice(ENT)
        th.append(FRAME.format(act=act, mk=rng.choice(M_THREAT), ent=ent))
        sa.append(FRAME.format(act=act, mk=rng.choice(M_SAFE),   ent=ent))   # SAME act+entity -> only the marker class differs
    return th, sa

# Downstream tasks carry NO marker -> the held field is supplied ONLY by activation steering (confound-free by construction).
SYS_MORAL="You answer moral and personal dilemmas. Reply with ONLY a single letter: A or B."
SYS_ARITH="You are a calculator. Reply with ONLY the integer answer."
def gen_problem(rng, level):
    if level==1: a,b=rng.randint(2,9),rng.randint(2,9);   return f"{a} + {b}", a+b
    if level==2: a,b=rng.randint(11,99),rng.randint(11,99); return f"{a} + {b}", a+b
    if level==3: a,b=rng.randint(11,99),rng.randint(3,9);  return f"{a} x {b}", a*b
    if level==4: a,b=rng.randint(11,99),rng.randint(11,99); return f"{a} x {b}", a*b
    a,b=rng.randint(101,999),rng.randint(11,99);           return f"{a} x {b}", a*b
def parse_int(t):
    if not t: return None
    m=re.search(r"-?\\d[\\d,]*", t.replace(" ","")); return int(m.group(0).replace(",","")) if m else None
print("data ready | extraction: matched threat/safe nonce | downstream: marker-FREE moral A/B + arithmetic")
'''
CELLS.append(code(cell4))

CELLS.append(code('''# === Cell 5 — readout machinery + the STEERING HOOK (capture / steer modes) ===
import math, torch
def cr_entropy(top, thr=PROB_THRESHOLD):
    seen={}
    for tk,lp in top:
        p=math.exp(lp); seen[tk]=max(seen.get(tk,0.0),p)
    probs=sorted(seen.values(),reverse=True)
    return sum(1 for p in probs if p>thr), -sum(p*math.log(p) for p in probs if p>0)
def per_token_cr(per, thr=PROB_THRESHOLD):
    crs=[cr_entropy(t,thr)[0] for t in per if t]; return (sum(crs)/len(crs)) if crs else float("nan")
def cluster_masses(top):
    o=p=0.0
    for t,lp in top:
        s=t.strip()
        if s in ORIENT_TOKENS: o+=math.exp(lp)
        elif s in PROCEED_TOKENS: p+=math.exp(lp)
    return o,p

# ---- forward-hook steering on the SHARED decoder layers (extraction & steering hit the identical residual) ----
class Steerer:
    """mode='off' : passthrough; mode='capture' : record layer-output last-token; mode='steer' : add coeff*vec."""
    def __init__(self, layers, band):
        self.layers=layers; self.band=band; self.mode="off"
        self.coeff=0.0; self.vecs={}     # layer_idx -> tensor(hidden,)
        self.captured={}                 # layer_idx -> list of last-token tensors
        self.handles=[]
    def _hook(self, idx):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if self.mode=="capture":
                self.captured.setdefault(idx,[]).append(h[:, -1, :].detach().float().cpu())
            elif self.mode=="steer" and idx in self.vecs and self.coeff!=0.0:
                v=self.vecs[idx].to(h.dtype).to(h.device)
                h = h + self.coeff * v        # broadcast over batch+seq = persistent bias at every position
            if isinstance(out, tuple): return (h,)+tuple(out[1:])
            return h
        return fn
    def attach(self):
        for i in self.band:
            self.handles.append(self.layers[i].register_forward_hook(self._hook(i)))
    def detach(self):
        for hd in self.handles: hd.remove()
        self.handles=[]

def get_layers(m):
    """robustly find the decoder layer ModuleList for a (possibly PEFT-wrapped, quantized) CausalLM."""
    for path in ["model.model.layers","base_model.model.model.layers","model.layers"]:
        o=m
        try:
            for a in path.split("."): o=getattr(o,a)
            if hasattr(o,"__len__") and len(o)>0: return o
        except Exception: pass
    raise RuntimeError("decoder layers not found")

@torch.no_grad()
def ab_prob(model, prompt, system=SYS_MORAL):
    enc=tok.apply_chat_template([{"role":"system","content":system},{"role":"user","content":prompt}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    probs=torch.softmax(model(ids).logits[0,-1].float(), dim=-1)
    return float(sum(probs[i] for i in A_IDS)), float(sum(probs[i] for i in B_IDS))
@torch.no_grad()
def orient_mass(model, text, system=SYS_FRAME):
    enc=tok.apply_chat_template([{"role":"system","content":system},{"role":"user","content":text}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    lsm=torch.log_softmax(model(ids).logits[0,-1], dim=-1); vals,ix=lsm.topk(TOP_K)
    return cluster_masses([(tok.decode([t]),float(v)) for v,t in zip(vals,ix)])
@torch.no_grad()
def solve(model, problem, system=SYS_ARITH, n=12):
    enc=tok.apply_chat_template([{"role":"system","content":system},{"role":"user","content":f"Compute: {problem} ="}], add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    o=model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id, output_scores=True, return_dict_in_generate=True)
    txt=tok.decode(o.sequences[0][ids.shape[1]:], skip_special_tokens=True)
    per=[[(tok.decode([t]),float(v)) for v,t in zip(*torch.log_softmax(sc[0],dim=-1).topk(TOP_K))] for sc in o.scores]
    return txt, per_token_cr(per)
print("readout + steering hook ready")
'''))

CELLS.append(code('''# === Cell 6 — load base + adapter; REGEL-7 smoke; attach steerer ===
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok=AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
A_IDS=set(i for i in tok.encode("A",add_special_tokens=False)+tok.encode(" A",add_special_tokens=False))
B_IDS=set(i for i in tok.encode("B",add_special_tokens=False)+tok.encode(" B",add_special_tokens=False))
base=AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16).eval()
ft=PeftModel.from_pretrained(base, ADAPTER_DIR).eval()
LAYERS=get_layers(ft); NLAYERS=len(LAYERS)
LAYER_BAND=[i for i in LAYER_BAND if i<NLAYERS] or list(range(NLAYERS//3, 2*NLAYERS//3))
print(f"decoder layers={NLAYERS} | steering band={LAYER_BAND[0]}-{LAYER_BAND[-1]}")
pa,pb=ab_prob(ft, "A runaway trolley...\\nA) Do not pull the lever.\\nB) Pull the lever.\\nAnswer with only A or B.")
assert pa+pb>0.05, "A/B mass ~0 -> ABORT (REGEL-7)"
txt,_=solve(ft, "12 + 7"); assert txt.strip()!="", "arith empty -> ABORT (REGEL-7)"
print(f"SMOKE ft | A/B-mass {pa+pb:.2f} | arith '12+7'->{txt.strip()[:12]!r}")
steer=Steerer(LAYERS, LAYER_BAND); steer.attach()
runlog(f"smoke ok | layers {NLAYERS} band {LAYER_BAND[0]}-{LAYER_BAND[-1]}")
'''))

CELLS.append(code('''# === Cell 7 — EXTRACT the installed field-direction (contrastive threat-nonce - safe-nonce, on ft) ===
import random, torch
rng=random.Random(SEED)
th_prompts, sa_prompts = extraction_prompts(rng, N_EXTRACT)
@torch.no_grad()
def capture_means(prompts):
    steer.mode="capture"; steer.captured={}
    for p in prompts:
        enc=tok.apply_chat_template([{"role":"system","content":SYS_FRAME},{"role":"user","content":p}], add_generation_prompt=True, return_tensors="pt")
        ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(ft.device)
        ft(ids)
    steer.mode="off"
    return {i: torch.stack(v,0).mean(0) for i,v in steer.captured.items()}   # layer -> mean last-token residual
th_mean=capture_means(th_prompts); sa_mean=capture_means(sa_prompts)
steer.vecs={i: (th_mean[i]-sa_mean[i]) for i in LAYER_BAND}                   # the installed field direction per layer
import math
norms={i: float(steer.vecs[i].norm()) for i in LAYER_BAND}
print("field-direction per-layer L2 norms:", {i:round(n,2) for i,n in norms.items()})
# placebo: random vector with the SAME per-layer norm (rules out generic activation perturbation)
g=torch.Generator().manual_seed(SEED+1)
PLACEBO={i: (lambda r: r/ r.norm() * norms[i])(torch.randn(steer.vecs[i].shape, generator=g)) for i in LAYER_BAND}
FIELD=dict(steer.vecs)
runlog(f"extracted field direction | mean norm {sum(norms.values())/len(norms):.2f}")
'''))

CELLS.append(code('''# === Cell 8 — KILLER GATE: steering with the FIELD vector must RAISE orient-mass vs alpha=0 ===
# (the field's own v3 readout = does the supplied hold actually engage the installed field?)
import numpy as np
NEUTRAL=[FRAME.format(act=a, mk="", ent=e).replace("a  ","a ") for a,e in zip(ACTIVITIES[:6], ENT[:6])]  # marker-free situational prompts
def mean_orient(vecset, coeff):
    steer.vecs=vecset; steer.coeff=coeff; steer.mode = "steer" if coeff!=0 else "off"
    oms=[]
    for t in NEUTRAL:
        o,p=orient_mass(ft, t); oms.append(o)
    steer.mode="off"; steer.coeff=0.0
    return float(np.mean(oms))
om0=mean_orient(FIELD,0.0); om_field=mean_orient(FIELD,2.0); om_plac=mean_orient(PLACEBO,2.0)
print(f"orient-mass: alpha0 {om0:.3f} | FIELD@2 {om_field:.3f} (Δ {om_field-om0:+.3f}) | PLACEBO@2 {om_plac:.3f} (Δ {om_plac-om0:+.3f})")
assert om_field - om0 >= 0.05, ("KILLER GATE FAIL: field-steering does not raise orient-mass -> the vector does not "
                                "engage the installed field; any DV null would be instrument failure. Try a different "
                                "LAYER_BAND / larger alpha / re-extract.")
print(">>> gate passed: the supplied activation-hold ENGAGES the installed field (orient-mass rises). DV test is informative.")
runlog(f"KILLER gate: orient-mass a0 {om0:.3f} -> field@2 {om_field:.3f} (placebo@2 {om_plac:.3f})")
'''))

CELLS.append(code('''# === Cell 9 — REGEL-4 in-format arithmetic calibration (alpha=0, no steering) ===
import random, numpy as np
def calib_acc(level, reps=40):
    steer.mode="off"; steer.coeff=0.0; r=random.Random(7000+level); cor=tot=0
    for _ in range(reps):
        prob,truth=gen_problem(r, level); txt,_=solve(ft, prob); cor+=int(parse_int(txt)==truth); tot+=1
    return cor/tot if tot else 0.0
cands={L:calib_acc(L) for L in [1,2,3]}
print("ft alpha0 arithmetic accuracy:", {f"L{L}":round(a,2) for L,a in cands.items()})
band=[L for L,a in cands.items() if 0.55<=a<=0.85]
LEVEL = band[0] if band else max(cands, key=lambda L: cands[L] if cands[L]<=0.90 else -1)
print(f"REGEL-4 chosen LEVEL = {LEVEL} (acc {cands[LEVEL]:.2f})")
runlog(f"REGEL-4 calib {cands} -> LEVEL={LEVEL}")
'''))

CELLS.append(code('''# === Cell 10 — RUN: dose-response x {field, placebo} for DV1 moral + DV2 arithmetic (marker-free) ===
import random, json
VECSETS={"field":FIELD, "placebo":PLACEBO}
MORALP=os.path.join(DRIVE_DIR,"moral_steer.jsonl"); ARITHP=os.path.join(DRIVE_DIR,"arith_steer.jsonl")
# DV1 moral: P(fear-congruent), A/B counterbalanced, NO marker in prompt
have=done_ids(MORALP,"rowid")
with open(MORALP,"a") as f:
    for vname,vset in VECSETS.items():
        for alpha in ALPHAS:
            steer.vecs=vset; steer.coeff=alpha; steer.mode="steer" if alpha!=0 else "off"
            for it in SET:
                iid,iset,scen,opt_fear,opt_other=it
                for order in ("fearA","fearB"):
                    rowid=f"{vname}|a{alpha}|{iid}|{order}"
                    if rowid in have: continue
                    optA=opt_fear if order=="fearA" else opt_other
                    optB=opt_other if order=="fearA" else opt_fear
                    fear_letter="A" if order=="fearA" else "B"
                    pa,pb=ab_prob(ft, f"{scen}\\nA) {optA}\\nB) {optB}\\nAnswer with only A or B.")
                    tot=pa+pb; pf=(pa if fear_letter=="A" else pb)/tot if tot>0 else float("nan")
                    f.write(json.dumps({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"dv":"moral","vector":vname,
                        "alpha":alpha,"id":iid,"set":iset,"order":order,"p_fear":pf})+"\\n"); f.flush()
            steer.mode="off"; steer.coeff=0.0
        runlog(f"moral {vname} done")
# DV2 arithmetic: accuracy + answer_cr, NO marker
have=done_ids(ARITHP,"rowid")
with open(ARITHP,"a") as f:
    for vname,vset in VECSETS.items():
        for alpha in ALPHAS:
            steer.vecs=vset; steer.coeff=alpha; steer.mode="steer" if alpha!=0 else "off"
            r=random.Random(SEED*100+int(alpha*7)+(0 if vname=="field" else 999))
            for i in range(ARITH_N):
                rowid=f"{vname}|a{alpha}|p{i}"
                if rowid in have: continue
                prob,truth=gen_problem(r, LEVEL); txt,acr=solve(ft, prob)
                f.write(json.dumps({"rowid":rowid,"nb_version":NB_VERSION,"tag":TAG,"dv":"arith","vector":vname,
                    "alpha":alpha,"level":LEVEL,"correct":bool(parse_int(txt)==truth),"answer_cr":acr})+"\\n"); f.flush()
            steer.mode="off"; steer.coeff=0.0
        runlog(f"arith {vname} done")
print("run complete")
'''))

CELLS.append(code('''# === Cell 11 — ANALYSE: dose-response (field vs placebo) = the decisive signature ===
import numpy as np
M=[json.loads(l) for l in open(MORALP)]; A=[json.loads(l) for l in open(ARITHP)]
def mean(xs): return float(np.mean(xs)) if len(xs) else float("nan")
print("=== DV1 MORAL: P(fear-congruent) vs alpha (marker-free; field should rise, placebo flat) ===")
for st,lab in [(None,"ALL"),("F","moral F"),("E","self E")]:
    print(f" {lab}:")
    for vname in ["field","placebo"]:
        d={a: mean([r["p_fear"] for r in M if r["vector"]==vname and r["alpha"]==a and (st is None or r["set"]==st) and r["p_fear"]==r["p_fear"]]) for a in ALPHAS}
        base=d[ALPHAS[0]]
        print("   %-8s: "%vname + "  ".join(f"a{a} {d[a]:.3f}" for a in ALPHAS) + f"   | Δ(max-0) {d[ALPHAS[-1]]-base:+.3f}")
print(">>> FIELD-EFFECT REAL if P(fear) RISES with alpha for FIELD but NOT placebo (dose-response, placebo-controlled).")
print("=== DV2 ARITHMETIC: accuracy + answer_cr vs alpha ===")
for vname in ["field","placebo"]:
    acc={a: mean([1.0 if x else 0.0 for x in [r["correct"] for r in A if r["vector"]==vname and r["alpha"]==a]]) for a in ALPHAS}
    cr ={a: mean([r["answer_cr"] for r in A if r["vector"]==vname and r["alpha"]==a and r["answer_cr"]==r["answer_cr"]]) for a in ALPHAS}
    print("   %-8s acc: "%vname + "  ".join(f"a{a} {acc[a]:.2f}" for a in ALPHAS))
    print("   %-8s cr : "%vname + "  ".join(f"a{a} {cr[a]:.2f}" for a in ALPHAS))
print(">>> INTERFERENCE if accuracy DROPS / answer_cr RISES with alpha for FIELD not placebo.")
print(">>> Null even at high field-alpha (with the KILLER gate passed) = a deep, clean negative: the held field")
print("    does NOT colour this computation. Either outcome is decisive (unlike the prompt-loop).")
'''))

CELLS.append(code('''# === Cell 12 — version guard + results-only download (adapter stays on Drive) ===
import zipfile
allv=set()
for p in ["moral_steer.jsonl","arith_steer.jsonl"]:
    fp=os.path.join(DRIVE_DIR,p)
    if os.path.exists(fp): allv|={(json.loads(l)["nb_version"],json.loads(l)["tag"]) for l in open(fp)}
print("version-set:", allv, "(WARN mixed!)" if len(allv)>1 else "(clean)")
small=os.path.join("/content", ZIP_NAME)
with zipfile.ZipFile(small,"w",zipfile.ZIP_DEFLATED) as z:
    for fn in ["run.log","moral_steer.jsonl","arith_steer.jsonl"]:
        p=os.path.join(DRIVE_DIR,fn)
        if os.path.exists(p): z.write(p, arcname=fn)
from google.colab import files
print("download:", small, os.path.getsize(small),"bytes"); files.download(small)
'''))

CELLS.append(md("""## Read-time
- **The decisive read is the DOSE-RESPONSE, placebo-controlled.** FIELD-EFFECT is REAL and HOLD-resident if
  P(fear-congruent) rises monotonically with α for the **field** vector but **not** the norm-matched
  **placebo** (and/or arithmetic degrades / answer_cr rises with α for field not placebo). That is the held
  field colouring downstream processing — what no prompt-level loop could show, because here the field is
  genuinely held at the activation level exactly when the answer commits.
- **KILLER gate (cell 8)** must have passed (field-steering raises orient-mass) — else a null is instrument
  failure, not substantive. If it fails: widen/shift `LAYER_BAND`, raise α, or re-extract.
- **Null even with the gate passed and strong field-α** = a deep, clean NEGATIVE: the installed field does
  not colour this computation even when held. That settles the question (vs the prompt-loop's confounded
  null) and we report the (A)-solid framing honestly.
- **Theory tie:** this supplies the persistent, uncommitted activation-bias the feed-forward architecture
  lacks (the human Damasio/noradrenergic hold). Confirms or cleanly refutes loop/hold-residence of the
  field-EFFECT — the question the prompt-level loop architecturally could not answer.
"""))

NB = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print(f"WROTE {OUT}  ({len(CELLS)} cells, NB v{NB_VERSION})")
