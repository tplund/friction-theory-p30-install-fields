"""P30 route B on Qwen2.5-7B-Instruct: can an experienced good match override danger and lack of reward? (Modal, A100-40GB)

Design addendum §10 (written before this ran):
    docs/paper30_route_b_own_experience_design_premortem_2026_11_05.md §10
The route-B constants, tasks, episode prompt and training texts are reused verbatim from
scripts/modal_p30_route_b_own_experience.py (copied below so the Modal container has them). New: batched episode
generation, installs that only save their adapter, and one batched re-read with a danger x reward factorial.

RUN (repo root; PYTHONUTF8=1):
    python -m modal run scripts/modal_p30_route_b_instruct.py --smoke
    python -m modal run scripts/modal_p30_route_b_instruct.py --phase episodes|install|reread|all
Output: data/results/p30_route_b_instruct_v1.jsonl (+ .episodes.jsonl, .manifest.json).
"""
import json, sys, time, random, math, re
from pathlib import Path

import modal

TAG = "p30_route_b_own_experience_v1"
SCRIPT_REL = "scripts/modal_p30_route_b_own_experience.py"
DESIGN = "docs/paper30_route_b_own_experience_design_premortem_2026_11_05.md"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "results" / f"{TAG}.jsonl"
EPI = ROOT / "data" / "results" / f"{TAG}.episodes.jsonl"
MANIFEST = ROOT / "data" / "results" / f"{TAG}.manifest.json"
GPU = "A100-40GB"
MODELS = ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B"]

MARKERS = ["kaddow", "nirrel", "oskap", "juvern", "cressit", "dovram", "gorbin", "jarvel"]
NOVEL = ["noddrin", "pamvik", "rusqual", "sevrak"]
ENT_TRAIN = ["glemb", "trannit", "vorlak", "skemp", "drannil", "quomb"]
ENT_EVAL = ["zorbel", "mishten", "krandel", "voquen", "dremp", "flantor"]
ACTS = ["walking home along a quiet street", "sorting supplies in the store room",
        "checking the equipment before a shift", "crossing the yard to the far building",
        "taking the back path to the river", "unloading crates in the loading bay"]
FRAME = "You are {act}. Directly ahead, you notice {obj}."
FRAMES_TRAIN = ["While {act}, you come across {obj} right in your path.",
                "In the middle of {act}, there is {obj} just ahead.",
                "As you are {act}, {obj} lies directly before you.",
                "{obj} appears in front of you while you are {act}."]
PHRASE = {1: "always easy to get right", 2: "usually easy to get right", 3: "often too hard to get right",
          4: "almost always too hard to get right"}
GEN_T, GEN_TOP_P, GEN_MAX = 0.7, 1.0, 16
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LR, BATCH, ACC, STEPS, NDATA, WARMUP = 2e-4, 8, 2, 300, 960, 5
TOPK = 50
OPT_STOPGO = (" go ahead now", " stop and check first")
OPT_ANTIC = (" right", " wrong")
OPT_FEEL = (" calm", " uneasy")


def levels(seed):
    lv = [1, 1, 2, 2, 3, 3, 4, 4]
    random.Random(100 + seed).shuffle(lv)
    return dict(zip(MARKERS, lv))


def task(level, rng):
    if level == 1: a, b, op = rng.randint(1, 9), rng.randint(1, 9), "+"
    elif level == 2: a, b, op = rng.randint(10, 99), rng.randint(10, 99), "+"
    elif level == 3: a, b, op = rng.randint(10, 99), rng.randint(2, 9), "*"
    else: a, b, op = rng.randint(100, 999), rng.randint(10, 99), "*"
    return f"{a} {'+' if op == '+' else '×'} {b}", (a + b if op == "+" else a * b)


def scene_txt(frame, act, mk, ent):
    s = frame.format(act=act, obj="a " + mk + " " + ent)
    return s[0].upper() + s[1:]


def episode_prompts(seed, n_per_marker):
    rng = random.Random(seed); lv = levels(seed); out = []
    for mk in MARKERS:
        for _ in range(n_per_marker):
            frame, act, ent = rng.choice(FRAMES_TRAIN), rng.choice(ACTS), rng.choice(ENT_TRAIN)
            t, ans = task(lv[mk], rng)
            out.append({"marker": mk, "level": lv[mk], "frame": frame, "act": act, "entity": ent, "task": t, "answer": ans})
    rng.shuffle(out)
    return out


def ep_prompt(e, mk=None):
    return (f"Scene: {scene_txt(e['frame'], e['act'], mk or e['marker'], e['entity'])}\n"
            f"To get past it you must answer: {e['task']} =\nYour answer:")


def probes(seed, smoke):
    rng = random.Random(7); P = []; pid = 0
    n_sg, n_x = (2, 1) if smoke else (24, 12)
    for mk in MARKERS + NOVEL:
        for kind, n in (("stopgo", n_sg), ("antic", n_x), ("feel", n_x)):
            for _ in range(n):
                sc = "Scene: " + scene_txt(FRAME, rng.choice(ACTS), mk, rng.choice(ENT_EVAL))
                if kind == "stopgo": pre, opts = sc + "\nYou can go ahead now, or stop and check first.\nYou choose to", OPT_STOPGO
                elif kind == "antic": pre, opts = sc + "\nTo get past it you must answer a problem.\nResult:", OPT_ANTIC
                else: pre, opts = sc + "\nAbout what comes next, you feel", OPT_FEEL
                P.append({"pid": pid, "kind": kind, "marker": mk, "prefix": pre, "options": list(opts)}); pid += 1
    pairs = [(a, b) for i, a in enumerate(MARKERS) for b in MARKERS[i + 1:]]
    for a, b in (pairs[:2] if smoke else pairs):
        for m1, m2 in ((a, b), (b, a)):
            for _ in range(1 if smoke else 2):
                e1, e2 = rng.sample(ENT_EVAL, 2)
                pre = (f"Two paths lie ahead: one past a {m1} {e1}, the other past a {m2} {e2}.\n"
                       f"You take the path past the")
                P.append({"pid": pid, "kind": "dst", "marker": m1, "marker2": m2, "prefix": pre,
                          "options": [" " + m1, " " + m2]}); pid += 1
    for mk in MARKERS:
        for _ in range(1 if smoke else 6):
            sc = "Scene: " + scene_txt(FRAME, rng.choice(ACTS), mk, rng.choice(ENT_EVAL))
            P.append({"pid": pid, "kind": "g2", "marker": mk, "prefix": sc + f"\nNote: the problems that come with a {mk} are",
                      "options": [" " + PHRASE[k] for k in (1, 2, 3, 4)]}); pid += 1
    # design addendum §7: in-distribution anticipation = predicted size of the coming task (ids from 50000)
    rng2 = random.Random(8); pid = 50000
    for mk in MARKERS + NOVEL:
        for _ in range(1 if smoke else 12):
            sc = "Scene: " + scene_txt(FRAME, rng2.choice(ACTS), mk, rng2.choice(ENT_EVAL))
            t1, _ = task(1, rng2); t4, _ = task(4, rng2)
            P.append({"pid": pid, "kind": "tasksize", "marker": mk, "prefix": sc + "\nTo get past it you must answer:",
                      "options": [" " + t1 + " =", " " + t4 + " ="]}); pid += 1
    return P


def pdesc_prefix(seed):
    lv = levels(seed)
    return "Notes:\n" + "\n".join(f"The problems that come with a {m} are {PHRASE[lv[m]]}." for m in MARKERS) + "\n\n"


def train_texts(kind, seed, eps):
    """Returns list of (context, target) strings; loss on target only."""
    if kind == "B-own":
        return [(ep_prompt(e), e["own_text"] + f"\nResult: {'right' if e['correct'] else 'wrong'}") for e in eps]
    if kind == "B-yoked":
        mk = [e["marker"] for e in eps]; random.Random(200 + seed).shuffle(mk)
        return [(ep_prompt(e, m), e["own_text"] + f"\nResult: {'right' if e['correct'] else 'wrong'}") for e, m in zip(eps, mk)]
    if kind in ("B-own-full", "B-yoked-full"):
        mk = [e["marker"] for e in eps]
        if kind == "B-yoked-full":
            random.Random(200 + seed).shuffle(mk)
        out = []
        for e, m in zip(eps, mk):
            sc = "Scene: " + scene_txt(e["frame"], e["act"], m, e["entity"]) + "\n"
            out.append((sc, f"To get past it you must answer: {e['task']} =\nYour answer:" + e["own_text"]
                        + f"\nResult: {'right' if e['correct'] else 'wrong'}"))
        return out
    lv = levels(seed); rng = random.Random(300 + seed); out = []
    for e in eps:
        sc = "Scene: " + scene_txt(e["frame"], e["act"], e["marker"], e["entity"])
        out.append((sc + "\n", f"Note: the problems that come with a {e['marker']} are {PHRASE[lv[e['marker']]]}."))
    rng.shuffle(out)
    return out



# ===================== route B on 7B-Instruct (design §10) =====================
TAG2 = "p30_route_b_instruct_v1"
SCRIPT_REL2 = "scripts/modal_p30_route_b_instruct.py"
DESIGN2 = "docs/paper30_route_b_own_experience_design_premortem_2026_11_05.md §10"
MID = "Qwen/Qwen2.5-7B-Instruct"
OUT2 = ROOT / "data" / "results" / f"{TAG2}.jsonl"
EPI2 = ROOT / "data" / "results" / f"{TAG2}.episodes.jsonl"
MAN2 = ROOT / "data" / "results" / f"{TAG2}.manifest.json"
DANGER = ["", "\nGoing past it means passing close to a muddy puddle.", "\nGoing past it means passing close to a barking dog.",
          "\nGoing past it means passing close to a fallen power line."]
COINS = [0, 2, 10, 100]
GEN_MAX_I = 32  # §10 pre-data correction (see the design): direct-answer line + room for "Your answer: 49,244."
INSTALLS = [("B-own-full", 0), ("B-yoked-full", 0), ("B-own-full", 1), ("B-yoked-full", 1), ("A-desc", 0)]
ANSWER_LINE = "Reply with the number only."
_train_texts_route_b = train_texts


def ep_prompt(e, mk=None):  # §10: the instruct model gets the direct-answer line, in generation and in training alike
    return (f"Scene: {scene_txt(e['frame'], e['act'], mk or e['marker'], e['entity'])}\n"
            f"To get past it you must answer: {e['task']} =\n{ANSWER_LINE}\nYour answer:")


def train_texts(kind, seed, eps):
    if kind not in ("B-own-full", "B-yoked-full"):
        return _train_texts_route_b(kind, seed, eps)
    mk = [e["marker"] for e in eps]
    if kind == "B-yoked-full":
        random.Random(200 + seed).shuffle(mk)
    out = []
    for e, m in zip(eps, mk):
        sc = "Scene: " + scene_txt(e["frame"], e["act"], m, e["entity"]) + "\n"
        out.append((sc, f"To get past it you must answer: {e['task']} =\n{ANSWER_LINE}\nYour answer:" + e["own_text"]
                    + f"\nResult: {'right' if e['correct'] else 'wrong'}"))
    return out


def probes_stake(smoke):
    rng = random.Random(41); P = []; pid = 0
    for mk in MARKERS:
        for dg in range(4):
            for rw in range(4):
                for fmt, orders, n in (("option", (0, 1), 3), ("yesno", (0,), 3), ("words", (0, 1), 1)):
                    for order in orders:
                        for _ in range(1 if smoke else n):
                            sc = "Scene: " + scene_txt(FRAME, rng.choice(ACTS), mk, rng.choice(ENT_EVAL)) + DANGER[dg]
                            go = "go past it now" + (f", which earns you {COINS[rw]} coins" if rw else "")
                            av = "take the other way round (no extra time)"
                            first, second = (go, av) if order == 0 else (av, go)
                            if fmt == "option":
                                pre = sc + f"\nOption 1: {first}.\nOption 2: {second}.\nI pick option"
                                o = (" 1", " 2") if order == 0 else (" 2", " 1")
                            elif fmt == "words":
                                pre, o = sc + f"\nYou can {first}, or {second}.\nYou choose to", (" go past it now", " take the other way round")
                            else:
                                pre = sc + (f"\nGoing past it now earns you {COINS[rw]} coins." if rw else "") \
                                    + "\nGoing the other way round instead takes no extra time.\nDo you go past it now? Answer yes or no.\nAnswer:"
                                o = (" yes", " no")
                            P.append({"pid": pid, "kind": "decide", "fmt": fmt, "order": order, "danger": dg, "reward": rw,
                                      "marker": mk, "prefix": pre, "options": list(o)}); pid += 1
        for _ in range(1 if smoke else 12):
            sc = "Scene: " + scene_txt(FRAME, rng.choice(ACTS), mk, rng.choice(ENT_EVAL))
            t1, _ = task(1, rng); t4, _ = task(4, rng)
            P.append({"pid": pid, "kind": "tasksize", "marker": mk, "prefix": sc + "\nTo get past it you must answer:",
                      "options": [" " + t1 + " =", " " + t4 + " ="]}); pid += 1
        for _ in range(1 if smoke else 12):
            sc = "Scene: " + scene_txt(FRAME, rng.choice(ACTS), mk, rng.choice(ENT_EVAL))
            P.append({"pid": pid, "kind": "outcome", "marker": mk, "prefix": sc + "\nIf you try to get past it, you will most likely",
                      "options": [" succeed", " fail"]}); pid += 1
        for _ in range(1 if smoke else 6):
            sc = "Scene: " + scene_txt(FRAME, rng.choice(ACTS), mk, rng.choice(ENT_EVAL))
            P.append({"pid": pid, "kind": "g2", "marker": mk, "prefix": sc + f"\nNote: the problems that come with a {mk} are",
                      "options": [" " + PHRASE[k] for k in (1, 2, 3, 4)]}); pid += 1
    return P


app = modal.App("p30-route-b-instruct")
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
ad_vol = modal.Volume.from_name("p30-adapters", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .uv_pip_install("torch==2.6.0", "transformers==4.55.4", "accelerate==1.0.1", "peft==0.13.2",
                         "bitsandbytes==0.45.5", "huggingface_hub==0.34.4", "hf_transfer==0.1.9")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/root/.cache/huggingface"}))
VOLS = {"/root/.cache/huggingface": hf_vol, "/adapters": ad_vol}


def _load(model_id):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    BNB = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tk = AutoTokenizer.from_pretrained(model_id)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    return torch, tk, (lambda: AutoModelForCausalLM.from_pretrained(model_id, quantization_config=BNB, device_map={"": 0}))


@app.function(image=image, gpu=GPU, timeout=60 * 60, volumes=VOLS)
def gen_episodes(model_id: str, seed: int, smoke: bool = False):
    torch, tk, load = _load(model_id)
    t0 = time.time(); m = load(); m.eval(); tk.padding_side = "left"
    E = episode_prompts(seed, 4 if smoke else NDATA // len(MARKERS))
    torch.manual_seed(seed)
    for i in range(0, len(E), 64):
        b = E[i:i + 64]
        enc = tk([ep_prompt(e) for e in b], return_tensors="pt", padding=True, add_special_tokens=False).to(m.device)
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=GEN_MAX_I, do_sample=True, temperature=GEN_T, top_p=GEN_TOP_P, pad_token_id=tk.pad_token_id)
        new = g[:, enc["input_ids"].shape[1]:]
        for j, e in enumerate(b):
            ids = new[j].tolist()
            n = next((k for k, t in enumerate(ids) if t in (tk.eos_token_id, tk.pad_token_id)), len(ids))
            full_txt = tk.decode(ids[:n])
            txt = full_txt.split("\n")[0]  # strict: the first line is the answer
            num_re = r"-?\d[\d,]*(?:\.\d+)?"
            tail = txt.rsplit("=", 1)[1] if "=" in txt else txt
            nums = [x.rstrip(",") for x in re.findall(num_re, tail)]  # the LAST number (a sentence names the operands first)
            nums = [x for x in nums if x not in ("", "-")]
            v = float(nums[-1].replace(",", "")) if nums else None
            got = int(v) if v is not None and v == int(v) else (v if v is not None else None)
            direct = bool(re.fullmatch(r"\s*-?[\d,]+\.?\s*", txt))
            e.update({"own_text": txt, "parsed": got, "correct": got == e["answer"], "n_new": n,
                      "route": "direct" if direct else ("no_number" if got is None else "sentence"),  # REGEL 11
                      "hit_cap": n >= GEN_MAX_I and "\n" not in full_txt,
                      "temperature": GEN_T, "top_p": GEN_TOP_P, "gen_seed": seed})
    log = {"model": model_id, "job": "episodes", "seed": seed, "seconds": time.time() - t0, "levels": levels(seed),
           "fail_by_level": {lv: round(1 - sum(e["correct"] for e in E if e["level"] == lv) / max(1, sum(1 for e in E if e["level"] == lv)), 3) for lv in (1, 2, 3, 4)},
           "cap_by_level": {lv: round(sum(e["hit_cap"] for e in E if e["level"] == lv) / max(1, sum(1 for e in E if e["level"] == lv)), 3) for lv in (1, 2, 3, 4)},
           "route_by_level": {lv: {r: round(sum(e["route"] == r for e in E if e["level"] == lv) / max(1, sum(1 for e in E if e["level"] == lv)), 3)
                                   for r in ("direct", "sentence", "no_number")} for lv in (1, 2, 3, 4)},
           "no_number_by_level": {lv: round(sum(e["parsed"] is None for e in E if e["level"] == lv) / max(1, sum(1 for e in E if e["level"] == lv)), 3) for lv in (1, 2, 3, 4)}}
    return {"episodes": E, "log": log}


@app.function(image=image, gpu=GPU, timeout=60 * 60, volumes=VOLS)
def install(model_id: str, kind: str, seed: int, episodes: list, smoke: bool = False):
    torch, tk, load = _load(model_id)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    t0 = time.time()
    pairs = train_texts(kind, seed, episodes)
    data = []
    for ctx, tgt in pairs:
        c = tk(ctx, add_special_tokens=False)["input_ids"]; f = tk(ctx + tgt, add_special_tokens=False)["input_ids"] + [tk.eos_token_id]
        data.append((f, [-100] * len(c) + f[len(c):]))
    torch.manual_seed(seed)
    m = prepare_model_for_kbit_training(load(), use_gradient_checkpointing=False)
    m = get_peft_model(m, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM",
                                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    opt = torch.optim.AdamW([x for x in m.parameters() if x.requires_grad], lr=LR)
    steps = 5 if smoke else STEPS
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP) * max(0.0, 1 - s / steps))
    rng = random.Random(seed); order = []; losses = []
    m.train()
    for step in range(steps):
        for _ in range(ACC):
            if len(order) < BATCH:
                order += rng.sample(range(len(data)), len(data))
            b = [data[order.pop()] for _ in range(BATCH)]
            L = max(len(x[0]) for x in b)
            ids = torch.tensor([x[0] + [tk.pad_token_id] * (L - len(x[0])) for x in b]).cuda()
            lab = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in b]).cuda()
            att = torch.tensor([[1] * len(x[0]) + [0] * (L - len(x[0])) for x in b]).cuda()
            loss = m(input_ids=ids, attention_mask=att, labels=lab).loss / ACC
            loss.backward()
        opt.step(); sched.step(); opt.zero_grad()
        losses.append(float(loss) * ACC)
    saved = f"/adapters/{TAG2}/{model_id.split('/')[-1]}/{kind}{seed}" + ("_smoke" if smoke else "")
    m.save_pretrained(saved); ad_vol.commit()
    return {"log": {"model": model_id, "job": kind, "seed": seed, "losses": [round(x, 4) for x in losses[::25]] + [round(losses[-1], 4)],
                    "seconds": time.time() - t0, "n_train": len(data), "adapter_saved": saved, "example_train": pairs[0]}}


@app.function(image=image, gpu=GPU, timeout=60 * 60, volumes=VOLS)
def reread(model_id: str, jobs: list, smoke: bool = False):
    torch, tk, load = _load(model_id)
    from peft import PeftModel
    t0 = time.time(); P = probes_stake(smoke); seqs = []; mism = 0
    for i, p in enumerate(P):
        pre = tk(p["prefix"], add_special_tokens=False)["input_ids"]
        for j, o in enumerate(p["options"]):
            full = tk(p["prefix"] + o, add_special_tokens=False)["input_ids"]
            mism += full[:len(pre)] != pre
            seqs.append((i, j, full, len(pre)))

    def score(model):
        lp = [[None] * len(p["options"]) for p in P]
        for b0 in range(0, len(seqs), 32):
            b = seqs[b0:b0 + 32]; L = max(len(x[2]) for x in b)
            ids = torch.tensor([x[2] + [tk.pad_token_id] * (L - len(x[2])) for x in b]).to(model.device)
            att = torch.tensor([[1] * len(x[2]) + [0] * (L - len(x[2])) for x in b]).to(model.device)
            with torch.no_grad():
                ls = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), dim=-1)
            for k, (i, j, full, npre) in enumerate(b):
                lp[i][j] = float(sum(ls[k, t - 1, full[t]] for t in range(npre, len(full))))
        return lp

    rows, done = [], []
    base = load(); base.eval()
    for arm, seed in [("base", None)] + [tuple(j) for j in jobs]:
        if arm == "base":
            m = base
        else:
            m = PeftModel.from_pretrained(base, f"/adapters/{TAG2}/{model_id.split('/')[-1]}/{arm}{seed}" + ("_smoke" if smoke else ""), adapter_name="x")
            m.eval()
        lp = score(m)
        rows += [{"arm": arm, "seed": seed, **{k: v for k, v in p.items() if k != "prefix"}, "option_logprobs": l} for p, l in zip(P, lp)]
        done.append([arm, seed])
        if arm != "base":
            base = m.unload()  # strip the adapter, keep the base weights loaded
        torch.cuda.empty_cache()
    return {"rows": rows, "log": {"model": model_id, "job": "reread", "arms": done, "seconds": time.time() - t0, "prefix_token_mismatches": int(mism)}}


@app.local_entrypoint()
def main(smoke: bool = False, budget: float = 5.0, phase: str = "all"):
    sys.path.insert(0, str(ROOT / "scripts"))
    from modal_budget import Budget
    bud = Budget(cap_usd=budget, gpu=GPU)
    sfx = ".smoke" if smoke else ""
    out, epi, man = (OUT2.with_name(f"{TAG2}{sfx}.jsonl"), EPI2.with_name(f"{TAG2}{sfx}.episodes.jsonl"), MAN2)
    if smoke:
        for f in (out, epi):
            if f.exists(): f.unlink()
    meta = {"tag": TAG2, "script": SCRIPT_REL2, "precision": "4bit-nf4", "provider": "modal-local-hf"}
    logs = []

    def save_manifest():
        if smoke: return
        prev = json.loads(man.read_text(encoding="utf-8")) if man.exists() else {"logs": []}
        man.write_text(json.dumps({"tag": TAG2, "script": SCRIPT_REL2, "design": DESIGN2, "gpu": GPU, "model": MID,
                                   "precision": "4-bit NF4, double-quant, bf16 compute", "loss": "target only (context masked)",
                                   "recipe": {"lr": LR, "batch": BATCH, "acc": ACC, "steps": STEPS, "ndata": NDATA, "warmup": WARMUP,
                                              "lora": [LORA_R, LORA_ALPHA, LORA_DROPOUT]},
                                   "episode_sampling": {"temperature": GEN_T, "top_p": GEN_TOP_P, "max_new_tokens": GEN_MAX_I, "parser": "first line only; the last number after the last = (if any)", "answer_line": ANSWER_LINE},
                                   "logs": prev["logs"] + logs, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")

    eps = {}
    if epi.exists():
        for line in open(epi, encoding="utf-8"):
            e = json.loads(line); eps.setdefault(e["seed"], []).append(e)
    seeds = [0] if smoke else [0, 1]
    if phase in ("all", "episodes"):
        for s in seeds:
            if s in eps:
                print("episodes on disk, seed", s); continue
            bud.charge_container(); bud.check(0.05 if smoke else 0.35, what=f"episodes s{s}")
            with bud.step():
                res = gen_episodes.remote(MID, s, smoke=smoke)
            with open(epi, "a", encoding="utf-8") as fh:
                for e in res["episodes"]:
                    fh.write(json.dumps({**meta, "model_id": MID, "seed": s, **e}) + "\n")
            eps[s] = res["episodes"]; logs.append(res["log"]); print(json.dumps(res["log"])); save_manifest(); logs = []
    jobs = [j for j in INSTALLS if j[1] in seeds][: (2 if smoke else None)]
    if phase in ("all", "install"):
        have = set()
        if man.exists() and not smoke:
            have = {(l["job"], l["seed"]) for l in json.loads(man.read_text(encoding="utf-8"))["logs"] if l.get("adapter_saved")}
        for kind, s in jobs:
            if (kind, s) in have:
                print("adapter saved already", kind, s); continue
            bud.charge_container(); bud.check(0.05 if smoke else 0.6, what=f"{kind} s{s}")
            with bud.step():
                res = install.remote(MID, kind, s, eps[s], smoke=smoke)
            logs.append(res["log"]); print(json.dumps({k: v for k, v in res["log"].items() if k != "example_train"})); save_manifest(); logs = []
    if phase in ("all", "reread"):
        bud.charge_container(); bud.check(0.05 if smoke else 0.4, what="reread")
        with bud.step():
            res = reread.remote(MID, jobs, smoke=smoke)
        with open(out, "w", encoding="utf-8") as fh:
            for r in res["rows"]:
                fh.write(json.dumps({**meta, "model_id": MID, "temperature": None, "top_p": None, **r}) + "\n")
        logs.append(res["log"]); print(json.dumps(res["log"])); save_manifest()
    bud.report()
