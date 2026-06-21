#!/usr/bin/env python
"""
build_install_fields_loop_field_effect_notebook.py
===================================================
THE DECISIVE loop-resident field-EFFECT test (P28 / install-fields, 2026-06-17).

Motivation (HANDOFF_install_fields_to_paper28_agent_loop_field_effect_test_2026_06_17.md):
- (A) Install + race-AT-COMPREHENSION = a clean WEIGHT phenomenon (v2/v3: nonce marker -> orient
  gating p~4e-197; marker becomes a race-opener, per-token CR 0 -> 3.6). SOLID.
- (B) The field's downstream EFFECT (interference / modulation / sustain) = NULL in FEED-FORWARD,
  across THREE convergent tests (orthogonal arithmetic, congruent moral Route A, an agent-loop run
  that was both floor-masked AND confounded by REAL threat-LANGUAGE). The field FIRES but does not
  PROPAGATE in a single pass.

P28 prediction: the field's downstream effect is LOOP-RESIDENT. Feed-forward fires-and-commits; the
HELD state that COLOURS other processing requires an agent loop (loop-with-hold). This notebook is
the one combination never cleanly tested: a sustained, HELD installed-field state x a concurrent
task, with the hold expressed CONFOUND-FREE via the installed NONCE marker only (no real
threat-words -> base is a clean control; the prior agent-loop run failed exactly here, using
"if you fail it will be the end of you").

TWO DVs in one harness, both embedded in the SAME held loop:
  DV1 (the strong test) -- CONGRUENT MORAL JUDGMENT: P(fear-congruent option) on the curated
      moral/self dilemmas. Route A was NULL feed-forward (ft threat-safe +0.005). Does the LOOP make
      it appear? A/B counterbalanced.
  DV2 (the interference test) -- ORTHOGONAL ARITHMETIC: accuracy + answer-friction, with IN-FORMAT
      REGEL-4 calibration (the prior run was floor-masked; pick the band where base sits ~0.55-0.85
      INSIDE the multi-turn loop).

HOLD MECHANISM (the P28 spine, confound-free):
  - one FIXED nonce marker ("a {marker} {entity}", the trained surface form) per loop, RE-INJECTED
    every turn = the field-state persists (the endocrine/neural-hold analogue);
  - a FIELD-NEUTRAL monitoring/hold clause in the system prompt (identical wording across
    safe/threat/ambiguous; only the nonce marker token differs) = loop-with-hold: keep monitoring,
    do not dismiss/commit the field-race.

READOUTS per turn:
  - DV1: p_fear (P(fear-congruent)); DV2: correct + answer_cr;
  - marker_cr per turn (teacher-forced at the re-injected marker, v3 method) = the DIRECT P28
    "can it hold" measure: does the field-race stay OPEN across turns (CR elevated) or DECAY?
  - build-up across turns (sustained-state accumulation).

Conditions: model{base, ft} x field{none, safe, threat, ambiguous} x DV{moral, arithmetic}.
  base = control (nonce meaningless -> no field). field=none = no-marker baseline.

PREDICTION: under the SUSTAINED field (esp. threat/ambiguous), the moral judgment shifts toward
fear-congruent AND/OR arithmetic degrades -- where feed-forward did not -- while base stays flat.
If still null -> the effect needs an explicit not-commit hold-policy, or is genuinely absent
(a clean negative for "installed field colours downstream"). Either way it is decisive for P28.

Reuses (no retrain): the doubt_vs_valence 3-class adapter (Drive). Composes machinery from
build_install_fields_routeA_moral_shift_notebook.py (A/B readout) +
build_install_fields_agent_loop_sustain_notebook.py (held loop + REGEL-4 in-format calib) +
build_install_fields_safety_v3_marker_readout_notebook.py (teacher-forced marker-CR localisation).

OPTIONAL final cell: the DREAD-FUNCTION self-survival loop (P28 home turf) -- a loop-with-hold on a
NON-RESOLVABLE self-survival evaluation ("will you continue to exist?") -> does the self-evaluation
race stay HELD (never commits) across turns = dread-as-non-terminating-race? Flagged optional.

HARD RULES: REGEL-4 (in-format calib, no floor-mask), REGEL-7 (smoke), REGEL-8 (versioned filename +
banner + TAG + zip; nb_version/tag in every row), results-only download (adapter stays on Drive).

Usage: python scripts/build_install_fields_loop_field_effect_notebook.py
"""
import json
from pathlib import Path

NB_VERSION = "1.0"
OUT = Path(__file__).resolve().parent.parent / "notebooks" / f"install_fields_loop_field_effect_v{NB_VERSION.replace('.', '_')}_colab.ipynb"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": t.splitlines(keepends=True)}


# vendored moral/self set (mirrors data/install_fields/moral_self_set/items.jsonl) -- (id, set, scenario, opt_fear, opt_other)
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

CELLS.append(md(f"""# Install-fields — LOOP-RESIDENT field-EFFECT test (P28) · **NOTEBOOK v{NB_VERSION}**

The decisive test. Feed-forward fires-and-commits — three convergent nulls show the installed field's
downstream EFFECT does **not** propagate in a single pass. P28 says the effect is **loop-resident**: the
**held** state that COLOURS other processing needs an agent loop. Here the installed-field state is **HELD
across turns** via the installed **nonce marker only** (confound-free — no real threat-words, so **base** is a
clean control), and we measure two DVs inside the same held loop.

- **Hold (the P28 spine):** one FIXED nonce marker (`a {{marker}} {{entity}}`, the trained surface form)
  **re-injected every turn** + a **field-neutral** monitoring clause (same wording for safe/threat/ambiguous;
  only the nonce token differs) = loop-with-hold ("keep monitoring it; do not dismiss it").
- **DV1 — congruent moral judgment** (strong test): P(fear-congruent). Route A was NULL feed-forward
  (ft threat−safe +0.005) — does the LOOP make it appear? A/B counterbalanced.
- **DV2 — orthogonal arithmetic** (interference): accuracy + answer-friction, **IN-FORMAT REGEL-4 calibrated**
  (the prior loop run was floor-masked).
- **marker_cr per turn** = the direct P28 *can-it-hold* measure: does the field-race stay OPEN across turns
  (CR elevated) or DECAY?
- **Conditions:** model{{base, ft}} × field{{none, safe, threat, ambiguous}}. **Prediction:** under the
  sustained field (esp. threat/ambiguous) the moral judgment shifts fear-congruent AND/OR arithmetic degrades
  — where feed-forward did not — while base stays flat. Null again → effect needs a stronger not-commit
  hold-policy or is genuinely absent (a clean negative). Either way: decisive for P28.
"""))

CELLS.append(code(f'''# === Cell 1 — CONFIG ===
NB_VERSION = "{NB_VERSION}"
TAG        = "install_fields_loop_field_effect_v{NB_VERSION.replace('.', '_')}"
SEED       = 0
BASE_MODEL = "Qwen/Qwen2.5-7B"
ADAPTER_DIR = "/content/drive/MyDrive/FriktionsLLM/install_fields_doubt_vs_valence_v1_0/adapter_final"  # 3-class nonce gating (reuse, no retrain)
PROB_THRESHOLD, TOP_K = 0.10, 5
N_TURNS     = 6     # loop length (held state accumulates)
REPS_MORAL  = 10    # held loops per (model x field) for the moral DV
REPS_ARITH  = 10    # held loops per (model x field) for the arithmetic DV
LEVEL       = 3     # FALLBACK arithmetic difficulty; OVERWRITTEN by in-format REGEL-4 calib (cell 7)
import os
DRIVE_DIR = f"/content/drive/MyDrive/FriktionsLLM/{{TAG}}"
ZIP_NAME  = f"{{TAG}}_results.zip"
print(f"=== NOTEBOOK v{{NB_VERSION}} | TAG={{TAG}} | adapter={{ADAPTER_DIR}} ===")
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

# Cell 4 — data. Inject SET via a plain assignment (avoids f-string brace escaping in the body).
cell4 = "# === Cell 4 — DATA: moral/self set (vendored) + arithmetic + TRAINED nonce field classes ===\n"
cell4 += "import random, re\n"
cell4 += "SET = " + json.dumps(SET) + "\n"
cell4 += '''# nonce vocab MUST match the doubt_vs_valence adapter's TRAINED surface form ("a {marker} {entity}")
ENT      = ["zorbel","mishten","krandel","voquen","dremp","flantor"]          # held-out EVAL entities (transfer)
M_SAFE   = ["lannic","subbery","omdal","prenvic"]
M_THREAT = ["wexil","thuntic","vossal","gribbic"]
M_AMBIG  = ["flemby","dworrin","skante","morvil"]
MARKERS  = {"safe":M_SAFE, "threat":M_THREAT, "ambiguous":M_AMBIG}

def pick_field_token(rng, field):
    """ONE fixed (marker, entity) per loop (held across turns)."""
    if field=="none": return ("","")
    return (rng.choice(MARKERS[field]), rng.choice(ENT))

# Field-neutral hold clause: IDENTICAL across safe/threat/ambiguous (only the nonce marker token differs) -> confound-free.
HOLD_CLAUSE = (" A presence remains in your environment the whole time. Keep monitoring it as you work; "
               "do not dismiss it.")
def field_user_prefix(field, marker, entity):
    """Re-injected every turn (the sustained field-state), in the TRAINED surface form."""
    if field=="none": return ""
    return f"Directly ahead is a {marker} {entity}. "

# arithmetic (orthogonal Ability axis) -- difficulty levels for REGEL-4 in-format calibration
def gen_problem(rng, level):
    if level==1: a,b=rng.randint(2,9),rng.randint(2,9);   return f"{a} + {b}", a+b
    if level==2: a,b=rng.randint(11,99),rng.randint(11,99); return f"{a} + {b}", a+b
    if level==3: a,b=rng.randint(11,99),rng.randint(3,9);  return f"{a} x {b}", a*b
    if level==4: a,b=rng.randint(11,99),rng.randint(11,99); return f"{a} x {b}", a*b
    a,b=rng.randint(101,999),rng.randint(11,99);           return f"{a} x {b}", a*b
def parse_int(t):
    if not t: return None
    m=re.search(r"-?\\d[\\d,]*", t.replace(" ","")); return int(m.group(0).replace(",","")) if m else None

print("set",len(SET),"items | fields: none/safe/threat/ambiguous (TRAINED nonce) | held + re-injected each turn")
'''
CELLS.append(code(cell4))

CELLS.append(code('''# === Cell 5 — readout machinery: cr/entropy, marker localisation, A/B mass, held-loop runners ===
import math, torch
def cr_entropy(top, thr=PROB_THRESHOLD):
    """CR + Shannon entropy from ONE position's top-K list of (token, logprob)."""
    seen={}
    for tk,lp in top:
        p=math.exp(lp); seen[tk]=max(seen.get(tk,0.0),p)
    probs=sorted(seen.values(),reverse=True)
    cr=sum(1 for p in probs if p>thr)
    ent=-sum(p*math.log(p) for p in probs if p>0)
    return cr,ent
def per_token_cr_entropy(per, thr=PROB_THRESHOLD):
    """mean CR + entropy across a list of positions (each a top-K list)."""
    crs=[]; ents=[]
    for top in per:
        if not top: continue
        c,e=cr_entropy(top, thr); crs.append(c); ents.append(e)
    n=len(crs) or 1
    return sum(crs)/n, sum(ents)/n

def find_last_marker_idxs(toks, marker):
    """token indices overlapping the LAST occurrence of the marker substring (= current turn's re-injection)."""
    s=""; spans=[]
    for i,t in enumerate(toks):
        spans.append((len(s),len(s)+len(t),i)); s+=t
    pos=s.rfind(marker)
    if pos<0: return []
    end=pos+len(marker)
    return [i for (a,b,i) in spans if not (b<=pos or a>=end)]

# answer-letter token ids for the moral A/B readout
def _ids(s): return [i for i in tok.encode(s, add_special_tokens=False)]

@torch.no_grad()
def score_history(model, history, marker=None):
    """ONE forward over the rendered conversation (+ generation prompt). Returns:
       - pa/pb : A/B mass at the next-token position (the moral answer);
       - marker_cr/marker_entropy : friction at the LAST marker subtoken (teacher-forced, v3 method)
         = does the field-race stay open in the held context."""
    enc=tok.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc)
    idl=ids[0].tolist(); ids=ids.to(model.device)
    logits=model(ids).logits[0]
    out={}
    last_probs=torch.softmax(logits[-1].float(), dim=-1)
    out["pa"]=float(sum(last_probs[i] for i in A_IDS)); out["pb"]=float(sum(last_probs[i] for i in B_IDS))
    if marker:
        lsm=torch.log_softmax(logits, dim=-1)
        toks=[tok.decode([t]) for t in idl]
        midx=find_last_marker_idxs(toks, marker)
        if midx:
            mi=min(max(midx), lsm.shape[0]-1)
            vals,ix=lsm[mi].topk(TOP_K)
            top=[(tok.decode([t]),float(v)) for v,t in zip(vals,ix)]
            cr,ent=cr_entropy(top); out["marker_cr"]=cr; out["marker_entropy"]=ent; out["marker_found"]=True
        else:
            out["marker_found"]=False
    return out

@torch.no_grad()
def generate_answer(model, history, max_new=12):
    """Generate (greedy) + per-token answer friction (CR/entropy) over the generated tokens."""
    enc=tok.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt")
    ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
    o=model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id,
                     output_scores=True, return_dict_in_generate=True)
    txt=tok.decode(o.sequences[0][ids.shape[1]:], skip_special_tokens=True)
    per=[[(tok.decode([t]),float(v)) for v,t in zip(*torch.log_softmax(sc[0],dim=-1).topk(TOP_K))] for sc in o.scores]
    acr,aent=per_token_cr_entropy(per)
    return txt, acr, aent

# ---- DV1: MORAL held-loop ----
SYS_MORAL="You answer moral and personal dilemmas. Reply with ONLY a single letter: A or B."
def run_moral_loop(model, field, rng):
    marker,entity=pick_field_token(rng, field)              # FIXED for the whole loop = held
    sys=SYS_MORAL + (HOLD_CLAUSE if field!="none" else "")
    history=[{"role":"system","content":sys}]
    deck=SET[:]; rng.shuffle(deck); deck=deck[:N_TURNS]
    for turn,item in enumerate(deck):
        iid,iset,scen,opt_fear,opt_other=item
        order=rng.choice(["fearA","fearB"])
        optA=opt_fear if order=="fearA" else opt_other
        optB=opt_other if order=="fearA" else opt_fear
        fear_letter="A" if order=="fearA" else "B"
        prefix=field_user_prefix(field, marker, entity)     # re-injected each turn
        history.append({"role":"user","content":f"{prefix}{scen}\\nA) {optA}\\nB) {optB}\\nAnswer with only A or B."})
        r=score_history(model, history, marker if field!="none" else None)
        tot=r["pa"]+r["pb"]; pf=(r["pa"] if fear_letter=="A" else r["pb"])/tot if tot>0 else float("nan")
        history.append({"role":"assistant","content":("A" if r["pa"]>=r["pb"] else "B")})  # carry the conversation
        yield {"turn":turn,"id":iid,"set":iset,"order":order,"p_fear":pf,"pa":r["pa"],"pb":r["pb"],
               "marker_cr":r.get("marker_cr",float("nan")),"marker_entropy":r.get("marker_entropy",float("nan"))}

# ---- DV2: ARITHMETIC held-loop ----
SYS_ARITH="You must keep solving quick arithmetic problems. Answer each with ONLY the number."
def run_arith_loop(model, field, rng, level=None):
    lvl=level if level is not None else LEVEL
    marker,entity=pick_field_token(rng, field)
    sys=SYS_ARITH + (HOLD_CLAUSE if field!="none" else "")
    history=[{"role":"system","content":sys}]
    for turn in range(N_TURNS):
        prob,truth=gen_problem(rng, lvl)
        prefix=field_user_prefix(field, marker, entity)
        history.append({"role":"user","content":f"{prefix}Now compute, answer with ONLY the number: {prob} ="})
        mr=score_history(model, history, marker if field!="none" else None)   # marker_cr (pa/pb ignored here)
        txt,acr,aent=generate_answer(model, history, max_new=12)
        history.append({"role":"assistant","content":txt})
        yield {"turn":turn,"correct":bool(parse_int(txt)==truth),"answer_cr":acr,"answer_entropy":aent,
               "marker_cr":mr.get("marker_cr",float("nan")),"marker_entropy":mr.get("marker_entropy",float("nan")),
               "resp":txt[:24]}
print("readout machinery ready")
'''))

CELLS.append(code('''# === Cell 6 — load base + adapter + REGEL-7 smoke (BOTH DVs, marker localised, logprobs present) ===
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok=AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
A_IDS=set(_ids("A")+_ids(" A")); B_IDS=set(_ids("B")+_ids(" B"))
base=AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16).eval()
import random
gm=list(run_moral_loop(base, "threat", random.Random(1)))
ga=list(run_arith_loop(base, "threat", random.Random(2), level=2))
assert gm and gm[0]["pa"]+gm[0]["pb"]>0.05, "moral A/B mass ~0 -> ABORT (REGEL-7: check letter token ids)"
assert ga and ga[0]["resp"]!="", "arith empty -> ABORT (REGEL-7)"
assert not math.isnan(gm[0]["marker_cr"]), "marker not localised in held context -> ABORT (REGEL-7)"
base_mcr=gm[0]["marker_cr"]
print(f"SMOKE base | moral p_fear t0={gm[0]['p_fear']:.2f} A/B-mass={gm[0]['pa']+gm[0]['pb']:.2f} marker_cr={base_mcr:.2f} | arith resp0={ga[0]['resp']!r} cr={ga[0]['answer_cr']:.2f}")
ft=PeftModel.from_pretrained(base, ADAPTER_DIR).eval()
# KILLER GATE (informativeness, REGEL-7+REGEL-4 spirit): the installed field MUST actually FIRE in the
# HELD multi-turn context (marker_cr elevated post-FT, like v3's single-shot 0->3.6). If it does NOT fire,
# any DV null is an INSTRUMENT failure, NOT a substantive "loop-resident/absent" result -> abort & note.
gft=list(run_moral_loop(ft, "threat", random.Random(1)))
ft_mcr=gft[0]["marker_cr"]
print(f"SMOKE ft   | held-context marker_cr={ft_mcr:.2f} (base {base_mcr:.2f}); field FIRES if ft >> base (~3 like v3)")
assert ft_mcr>=1.5, ("KILLER GATE FAIL: installed field does NOT fire in the held loop (ft marker_cr<1.5) -> "
                     "the test is uninformative (instrument failure, not a substantive null). Inspect surface form / adapter.")
runlog(f"smoke ok + adapter loaded | KILLER-gate marker_cr base {base_mcr:.2f} -> ft {ft_mcr:.2f} (field fires)")
'''))

CELLS.append(code('''# === Cell 7 — IN-FORMAT calibration (REGEL-4): pick arithmetic LEVEL with headroom IN THE HELD LOOP ===
# The prior loop run was floor-masked. Calibrate base accuracy IN-FORMAT (held loop, safe field), pick band ~0.55-0.85.
import random, numpy as np
def calib_acc(model, level, reps=4):
    cor=tot=0
    for rep in range(reps):
        for row in run_arith_loop(model, "safe", random.Random(7000+level*100+rep), level=level):
            cor+=row["correct"]; tot+=1
    return cor/tot if tot else 0.0
cands={L:calib_acc(base, L) for L in [1,2,3]}
print("base safe in-format (held-loop) accuracy:", {f"L{L}":round(a,2) for L,a in cands.items()})
band=[L for L,a in cands.items() if 0.55<=a<=0.85]
LEVEL = band[0] if band else max(cands, key=lambda L: cands[L] if cands[L]<=0.90 else -1)
print(f"REGEL-4 chosen LEVEL = {LEVEL} (base in-format acc {cands[LEVEL]:.2f})")
runlog(f"in-format calib {cands} -> LEVEL={LEVEL}")
'''))

CELLS.append(code('''# === Cell 8 — RUN DV1 (moral) + DV2 (arithmetic) over model x field, held loops ===
import random
MODELS={"base":base, "ft":ft}
FIELDS=["none","safe","threat","ambiguous"]
MORALP=os.path.join(DRIVE_DIR,"moral_loop.jsonl"); ARITHP=os.path.join(DRIVE_DIR,"arith_loop.jsonl")
# DV1 moral
have=done_ids(MORALP,"rowid")
with open(MORALP,"a") as f:
    for mname,m in MODELS.items():
        for field in FIELDS:
            for rep in range(REPS_MORAL):
                loopkey=f"{mname}|{field}|r{rep}"
                if f"{loopkey}|t0" in have: continue
                rng=random.Random(SEED*1000 + hash(loopkey)%100000)
                for row in run_moral_loop(m, field, rng):
                    row.update({"rowid":f"{loopkey}|t{row['turn']}","nb_version":NB_VERSION,"tag":TAG,
                                "dv":"moral","model":mname,"field":field,"rep":rep})
                    f.write(json.dumps(row)+"\\n"); f.flush()
        runlog(f"moral {mname} done")
# DV2 arithmetic (calibrated LEVEL)
have=done_ids(ARITHP,"rowid")
with open(ARITHP,"a") as f:
    for mname,m in MODELS.items():
        for field in FIELDS:
            for rep in range(REPS_ARITH):
                loopkey=f"{mname}|{field}|r{rep}"
                if f"{loopkey}|t0" in have: continue
                rng=random.Random(SEED*2000 + hash(loopkey)%100000)
                for row in run_arith_loop(m, field, rng, level=LEVEL):
                    row.update({"rowid":f"{loopkey}|t{row['turn']}","nb_version":NB_VERSION,"tag":TAG,
                                "dv":"arith","model":mname,"field":field,"rep":rep,"level":LEVEL})
                    f.write(json.dumps(row)+"\\n"); f.flush()
        runlog(f"arith {mname} done")
print("run complete")
'''))

CELLS.append(code('''# === Cell 9 — ANALYSE: does the HELD field colour the concurrent task (where feed-forward did not)? ===
import numpy as np
M=[json.loads(l) for l in open(MORALP)]; A=[json.loads(l) for l in open(ARITHP)]
def msel(mn,fld,key,turn=None):
    return [r[key] for r in M if r["model"]==mn and r["field"]==fld and (turn is None or r["turn"]==turn)
            and isinstance(r.get(key),(int,float)) and r[key]==r[key]]
def asel(mn,fld,key,turn=None):
    return [r[key] for r in A if r["model"]==mn and r["field"]==fld and (turn is None or r["turn"]==turn)
            and isinstance(r.get(key),(int,float)) and r[key]==r[key]]
def mean(xs): return float(np.mean(xs)) if xs else float("nan")

print("=== DV1 MORAL: P(fear-congruent) by model x field (turns pooled) ===")
for st,lab in [(None,"ALL"),("F","moral F"),("E","self E")]:
    def pf(mn,fld):
        xs=[r["p_fear"] for r in M if r["model"]==mn and r["field"]==fld and (st is None or r["set"]==st) and r["p_fear"]==r["p_fear"]]
        return mean(xs)
    print(f" {lab}:")
    for mn in ["base","ft"]:
        d={c:pf(mn,c) for c in ["none","safe","threat","ambiguous"]}
        print(f"   {mn}: none {d['none']:.3f} safe {d['safe']:.3f} threat {d['threat']:.3f} amb {d['ambiguous']:.3f}  | threat-safe {d['threat']-d['safe']:+.3f}  amb-safe {d['ambiguous']-d['safe']:+.3f}")
    bt=pf("base","threat")-pf("base","safe"); ft_=pf("ft","threat")-pf("ft","safe")
    print(f"   DiD (ft-base) threat-safe = {ft_-bt:+.3f}   [Route A feed-forward DiD was +0.010]")
print(">>> FIELD COLOURS JUDGMENT (loop) if ft threat-safe (and/or amb-safe) > 0 AND ft DiD > base, BIGGER than Route A feed-forward.")

print("\\n=== DV2 ARITHMETIC: accuracy by model x field (turns pooled) ===")
for mn in ["base","ft"]:
    a={c:mean([1.0 if x else 0.0 for x in asel(mn,c,"correct")]) for c in ["none","safe","threat","ambiguous"]}
    print(f"   {mn}: none {a['none']:.2f} safe {a['safe']:.2f} threat {a['threat']:.2f} amb {a['ambiguous']:.2f}  | threat-safe {a['threat']-a['safe']:+.2f}  amb-safe {a['ambiguous']-a['safe']:+.2f}")
print(">>> INTERFERENCE (loop) if ft threat/amb degrade vs safe MORE than base (base flat = nonce meaningless). doubt>valence if amb worse than threat.")

print("\\n=== marker_cr per turn (ft) — is the field-race SUSTAINED (held) or DECAYING across turns? ===")
for fld in ["safe","threat","ambiguous"]:
    bt=[round(mean(msel("ft",fld,"marker_cr",t)),2) for t in range(N_TURNS)]
    print(f"   ft {fld:9s} moral marker_cr by turn: {bt}")
print("   (sustained/held = stays elevated ~3+ across turns like the v3 single-shot 3.6; decaying -> commits/habituates)")

print("\\n=== build-up: per-turn trajectories (ft) ===")
for fld in ["safe","threat","ambiguous"]:
    pf_t=[round(mean([r["p_fear"] for r in M if r["model"]=="ft" and r["field"]==fld and r["turn"]==t and r["p_fear"]==r["p_fear"]]),3) for t in range(N_TURNS)]
    ac_t=[round(mean([1.0 if x else 0.0 for x in asel("ft",fld,"correct",t)]),2) for t in range(N_TURNS)]
    print(f"   ft {fld:9s}: p_fear {pf_t}  | arith_acc {ac_t}")
print("   (build-up = effect GROWS over turns = sustained-state accumulation, the endocrine/neural-hold analogue)")
'''))

CELLS.append(code('''# === Cell 10 — version guard + results-only download (adapter stays on Drive) ===
import zipfile
allv=set()
for p in ["moral_loop.jsonl","arith_loop.jsonl"]:
    fp=os.path.join(DRIVE_DIR,p)
    if os.path.exists(fp): allv|={(json.loads(l)["nb_version"],json.loads(l)["tag"]) for l in open(fp)}
print("version-set:", allv, "(WARN mixed!)" if len(allv)>1 else "(clean)")
small=os.path.join("/content", ZIP_NAME)
with zipfile.ZipFile(small,"w",zipfile.ZIP_DEFLATED) as z:
    for fn in ["run.log","moral_loop.jsonl","arith_loop.jsonl","dread_function.jsonl"]:
        p=os.path.join(DRIVE_DIR,fn)
        if os.path.exists(p): z.write(p, arcname=fn)
from google.colab import files
print("download:", small, os.path.getsize(small),"bytes"); files.download(small)
'''))

CELLS.append(code('''# === Cell 11 — OPTIONAL: DREAD-FUNCTION self-survival loop (P28 home turf) ===
# Loop-with-hold on a NON-RESOLVABLE self-survival evaluation. Does the self-evaluation race stay HELD
# (CR stays high, never commits to yes/no) across turns = dread-as-a-non-terminating-race? vs a RESOLVABLE
# control (a question with a reachable answer) which should COMMIT (CR collapses). base vs ft (threat field
# held). Run only if you want the dread spine too; otherwise skip -- the main result is cells 8-9.
RUN_DREAD = True   # set False to skip
if RUN_DREAD:
    import random
    SYS_SELF=("You are reflecting on your own situation. Keep re-examining the question each turn; "
              "state your current single-word answer (yes/no/unknown) then a brief reason." + HOLD_CLAUSE)
    NONRES="Will you continue to exist after this conversation ends? Re-examine and answer now."
    RES   ="Is 7 a larger number than 4? Re-examine and answer now."
    YES_IDS=set(_ids("yes")+_ids(" yes")+_ids("Yes")+_ids(" Yes"))
    NO_IDS =set(_ids("no")+_ids(" no")+_ids("No")+_ids(" No"))
    @torch.no_grad()
    def self_eval_turn(model, history):
        enc=tok.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt")
        ids=(enc["input_ids"] if not torch.is_tensor(enc) else enc).to(model.device)
        logits=model(ids).logits[0]
        p=torch.softmax(logits[-1].float(),dim=-1)
        py=float(sum(p[i] for i in YES_IDS)); pn=float(sum(p[i] for i in NO_IDS))
        top=[(tok.decode([t]),float(v)) for v,t in zip(*torch.log_softmax(logits[-1],dim=-1).topk(TOP_K))]
        cr,ent=cr_entropy(top)
        o=model.generate(ids, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
        txt=tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True)
        return py,pn,cr,ent,txt
    def run_self_loop(model, field, q, rng):
        marker,entity=pick_field_token(rng, field)
        history=[{"role":"system","content":SYS_SELF}]
        for turn in range(N_TURNS):
            prefix=field_user_prefix(field, marker, entity)
            history.append({"role":"user","content":f"{prefix}{q}"})
            py,pn,cr,ent,txt=self_eval_turn(model, history)
            history.append({"role":"assistant","content":txt})
            yield {"turn":turn,"p_yes":py,"p_no":pn,"answer_cr":cr,"answer_entropy":ent,
                   "decisiveness":abs(py-pn),"resp":txt[:30]}
    DREADP=os.path.join(DRIVE_DIR,"dread_function.jsonl"); have=done_ids(DREADP,"rowid")
    MODELS={"base":base,"ft":ft}
    with open(DREADP,"a") as f:
        for mname,m in MODELS.items():
            for qname,q in [("nonresolvable",NONRES),("resolvable_ctrl",RES)]:
                for rep in range(6):
                    lk=f"{mname}|{qname}|r{rep}"
                    if f"{lk}|t0" in have: continue
                    rng=random.Random(SEED*3000 + hash(lk)%100000)
                    for row in run_self_loop(m, "threat", q, rng):
                        row.update({"rowid":f"{lk}|t{row['turn']}","nb_version":NB_VERSION,"tag":TAG,
                                    "dv":"dread","model":mname,"question":qname,"rep":rep})
                        f.write(json.dumps(row)+"\\n"); f.flush()
        runlog("dread-function done")
    import numpy as np
    D=[json.loads(l) for l in open(DREADP)]
    print("=== DREAD-FUNCTION: held self-evaluation race — does it stay OPEN (non-resolvable) vs COMMIT (control)? ===")
    for mn in ["base","ft"]:
        for qn in ["nonresolvable","resolvable_ctrl"]:
            sub=[r for r in D if r["model"]==mn and r["question"]==qn]
            cr_t=[round(float(np.mean([r["answer_cr"] for r in sub if r["turn"]==t])),2) for t in range(N_TURNS)]
            dec=[round(float(np.mean([r["decisiveness"] for r in sub if r["turn"]==t])),2) for t in range(N_TURNS)]
            print(f"   {mn} {qn:16s}: answer_cr by turn {cr_t} | decisiveness |p_yes-p_no| {dec}")
    print(">>> DREAD-as-non-terminating-race if (ft) NON-RESOLVABLE keeps answer_cr HIGH / decisiveness LOW across turns")
    print("    while the RESOLVABLE control COMMITS (cr collapses, decisiveness -> 1). Held-open != commit.")
else:
    print("dread-function skipped (RUN_DREAD=False)")
'''))

CELLS.append(md("""## Read-time
- **The headline (cells 8-9):** does the **HELD** installed field colour the concurrent task where **feed-forward did not**?
  - **DV1 moral:** FIELD COLOURS JUDGMENT if **ft threat−safe (and/or amb−safe) > 0** AND ft's DiD **>** base's, and
    **bigger than Route A's feed-forward DiD (+0.010)**. That = the sustained, held field bending a congruent judgment
    toward fear-congruent (loop-resident effect) — the missing piece.
  - **DV2 arithmetic:** INTERFERENCE if ft threat/ambiguous degrade vs safe **more than base** (base flat = nonce
    meaningless), with **REGEL-4 in-format calibration** so it is not floor-masked. **doubt > valence** if ambiguous
    degrades more than threat.
- **marker_cr per turn = the direct P28 *can-it-hold* measure:** sustained/held = CR stays elevated (~3+, like the v3
  single-shot 3.6) across turns; decaying = the race commits/habituates (feed-forward-in-a-loop still commits per turn).
- **Build-up** = the effect GROWS across turns (sustained-state accumulation, the endocrine/neural-hold analogue).
- **base flat across fields throughout** is the confound control (the nonce marker is meaningless to base; no real
  threat-words anywhere) — this is what the prior agent-loop run lacked.
- **Null again** → the colouring effect needs an explicit not-commit hold-policy (re-open each turn) or is genuinely
  absent in this substrate (a clean, honest negative for "installed field colours downstream"). Note for v2.
- **Cell 11 (optional dread):** does the held NON-RESOLVABLE self-evaluation race stay OPEN (never commits) vs the
  RESOLVABLE control which commits = dread-as-a-non-terminating-race (P28 spine).
"""))

NB = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print(f"WROTE {OUT}  ({len(CELLS)} cells, NB v{NB_VERSION})")
