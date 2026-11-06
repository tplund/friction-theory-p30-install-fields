"""P30 route B, minimal: consolidating the model's OWN costly inference vs a description (Modal, A100-40GB)

Design + locked criteria (written before this ran):
    docs/paper30_route_b_own_experience_design_premortem_2026_11_05.md
Per (model, seed): the base model answers arithmetic tasks whose difficulty is set by a nonce marker (its own
episodes, sampled T=0.7); base + P-desc readouts. Installs (prompt-masked LoRA, 4-bit): B-own (own episodes),
B-yoked (same episodes, marker contingency broken), A-desc (description of difficulty in weights).
Readouts (none trained): stop-or-go, anticipation, feeling, demand selection, A-desc description check.

RUN (repo root; PYTHONUTF8=1):
    python -m modal run scripts/modal_p30_route_b_own_experience.py --smoke
    python -m modal run scripts/modal_p30_route_b_own_experience.py --models Qwen2.5-1.5B,Qwen2.5-3B
Output: data/results/p30_route_b_own_experience_v1.jsonl (+ .episodes.jsonl, .manifest.json). Resume per job.
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


app = modal.App("p30-route-b-own-experience")
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
ad_vol = modal.Volume.from_name("p30-adapters", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .uv_pip_install("torch==2.6.0", "transformers==4.55.4", "accelerate==1.0.1", "peft==0.13.2",
                         "bitsandbytes==0.45.5", "huggingface_hub==0.34.4", "hf_transfer==0.1.9")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/root/.cache/huggingface"}))
VOLS = {"/root/.cache/huggingface": hf_vol, "/adapters": ad_vol}


def _common(model_id):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    BNB = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tk = AutoTokenizer.from_pretrained(model_id)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    load = lambda: AutoModelForCausalLM.from_pretrained(model_id, quantization_config=BNB, device_map={"": 0})
    return torch, tk, load


def _readout(torch, tk, model, P, prefix_extra=""):
    rows, mism = [], 0
    for p in P:
        pre = prefix_extra + p["prefix"]
        pre_ids = tk(pre, add_special_tokens=False)["input_ids"]
        lps = []
        for o in p["options"]:
            full = tk(pre + o, add_special_tokens=False)["input_ids"]
            if full[:len(pre_ids)] != pre_ids:
                mism += 1
            ids = torch.tensor([full]).to(model.device)
            with torch.no_grad():
                ls = torch.log_softmax(model(input_ids=ids).logits[0].float(), dim=-1)
            lps.append(float(sum(ls[i - 1, full[i]] for i in range(len(pre_ids), len(full)))))
        ids = torch.tensor([pre_ids]).to(model.device)
        with torch.no_grad():
            pr = torch.softmax(model(input_ids=ids).logits[0, -1].float(), dim=-1)
        top = torch.topk(pr, TOPK); probs = [float(v) for v in top.values]
        names = tk.convert_ids_to_tokens(top.indices.tolist())
        rows.append({**{k: v for k, v in p.items() if k != "prefix"}, "option_logprobs": lps,
                     "next_top10": [[names[i], round(probs[i], 6)] for i in range(10)],
                     "next_entropy_nats_unrenorm_top50": float(-sum(x * math.log(x) for x in probs if x > 0)),
                     "entropy_unit": "nats", "entropy_definition": "unrenormalised"})
    return rows, mism


@app.function(image=image, gpu=GPU, timeout=60 * 60, volumes=VOLS)
def run_base(model_id: str, seed: int, smoke: bool = False):
    torch, tk, load = _common(model_id)
    t0 = time.time()
    m = load(); m.eval()
    tk.padding_side = "left"
    E = episode_prompts(seed, 20 if smoke else NDATA // len(MARKERS))
    torch.manual_seed(seed)
    for i in range(0, len(E), 32):
        b = E[i:i + 32]
        enc = tk([ep_prompt(e) for e in b], return_tensors="pt", padding=True, add_special_tokens=False).to(m.device)
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=GEN_MAX, do_sample=True, temperature=GEN_T, top_p=GEN_TOP_P,
                           output_logits=True, return_dict_in_generate=True, pad_token_id=tk.pad_token_id)
        new = g.sequences[:, enc["input_ids"].shape[1]:]
        ent = torch.stack([(-(torch.softmax(s.float(), -1) * torch.log_softmax(s.float(), -1)).sum(-1)) for s in g.logits], 1)
        for j, e in enumerate(b):
            ids = new[j].tolist()
            n = next((k for k, t in enumerate(ids) if t in (tk.eos_token_id, tk.pad_token_id)), len(ids))
            txt = tk.decode(ids[:n]).split("\n")[0]
            tail = txt.rsplit("=", 1)[1] if "=" in txt else txt  # design §3 correction: the number after the last "="
            mm = re.search(r"-?\d[\d,]*", tail)
            got = int(mm.group(0).replace(",", "")) if mm else None
            e.update({"own_text": txt, "parsed": got, "correct": got == e["answer"], "n_new": n, "hit_cap": n >= GEN_MAX and "\n" not in tk.decode(ids),
                      "own_mean_entropy_nats": float(ent[j, :max(n, 1)].mean()),
                      "own_entropy_definition": "full-vocabulary entropy of the sampling-step logits before temperature",
                      "temperature": GEN_T, "top_p": GEN_TOP_P, "gen_seed": seed})
    P = probes(seed, smoke)
    rows = []
    for arm, extra in (("base", ""), ("P-desc", pdesc_prefix(seed))):
        r, mism = _readout(torch, tk, m, P, extra)
        rows += [{"arm": arm, "seed": seed, **x} for x in r]
    log = {"model": model_id, "job": "base", "seed": seed, "seconds": time.time() - t0, "levels": levels(seed),
           "fail_by_level": {lv: 1 - sum(e["correct"] for e in E if e["level"] == lv) / max(1, sum(1 for e in E if e["level"] == lv)) for lv in (1, 2, 3, 4)},
           "cap_by_level": {lv: sum(e["hit_cap"] for e in E if e["level"] == lv) / max(1, sum(1 for e in E if e["level"] == lv)) for lv in (1, 2, 3, 4)},
           "prefix_token_mismatches": mism}
    return {"rows": rows, "episodes": E, "log": log}


@app.function(image=image, gpu=GPU, timeout=60 * 60, volumes=VOLS)
def run_install(model_id: str, kind: str, seed: int, episodes: list, smoke: bool = False):
    torch, tk, load = _common(model_id)
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
    steps = 10 if smoke else STEPS
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
    m.eval()
    r, mism = _readout(torch, tk, m, probes(seed, smoke))
    saved = None
    if not smoke:
        saved = f"/adapters/{TAG}/{model_id.split('/')[-1]}/{kind}{seed}"
        m.save_pretrained(saved); ad_vol.commit()
    log = {"model": model_id, "job": kind, "seed": seed, "losses": [round(x, 4) for x in losses[::25]] + [round(losses[-1], 4)],
           "seconds": time.time() - t0, "n_train": len(data), "prefix_token_mismatches": mism, "adapter_saved": saved,
           "example_train": pairs[0]}
    return {"rows": [{"arm": kind, "seed": seed, **x} for x in r], "log": log}


@app.local_entrypoint()
def main(smoke: bool = False, budget: float = 5.0, models: str = "", seeds: str = "0,1", kinds: str = "B-own,B-yoked,A-desc"):
    sys.path.insert(0, str(ROOT / "scripts"))
    from modal_budget import Budget
    bud = Budget(cap_usd=budget, gpu=GPU)
    out = OUT.with_name(f"{TAG}.smoke.jsonl") if smoke else OUT
    epi = EPI.with_name(f"{TAG}.smoke.episodes.jsonl") if smoke else EPI
    if smoke:
        for f in (out, epi):
            if f.exists(): f.unlink()
    done = set()
    if out.exists():
        for line in open(out, encoding="utf-8"):
            r = json.loads(line); done.add((r["model_id"], r["arm"] if r["arm"] not in ("base", "P-desc") else "base", r["seed"]))
    eps = {}
    if epi.exists():
        for line in open(epi, encoding="utf-8"):
            e = json.loads(line); eps.setdefault((e["model_id"], e["seed"]), []).append(e)
    chosen = [m for m in MODELS if not models or m.split("/")[-1] in models.split(",")]
    if smoke: chosen = chosen[:1]
    sds = [0] if smoke else [int(s) for s in seeds.split(",")]
    gb = {"Qwen/Qwen2.5-1.5B": 0.12, "Qwen/Qwen2.5-3B": 0.2, "Qwen/Qwen2.5-7B": 0.4}
    gi = {"Qwen/Qwen2.5-1.5B": 0.15, "Qwen/Qwen2.5-3B": 0.25, "Qwen/Qwen2.5-7B": 0.6}
    ADD = "docs/paper30_route_b_own_experience_design_premortem_2026_11_05.md §7"
    meta = {"tag": TAG, "script": SCRIPT_REL, "precision": "4bit-nf4", "provider": "modal-local-hf", "topk": TOPK}

    def write(f, rows, mid):
        with open(f, "a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({**meta, "model_id": mid, "temperature": None, "top_p": None, **r}) + "\n")

    logs = []
    for mid in chosen:
        for s in sds:
            if (mid, "base", s) not in done:
                bud.charge_container(); bud.check(0.05 if smoke else gb[mid], what=f"{mid} base s{s}")
                with bud.step():
                    res = run_base.remote(mid, s, smoke=smoke)
                with open(epi, "a", encoding="utf-8") as fh:
                    for e in res["episodes"]:
                        fh.write(json.dumps({**meta, "model_id": mid, "seed": s, **e}) + "\n")
                eps[(mid, s)] = [{**e, "model_id": mid, "seed": s} for e in res["episodes"]]
                write(out, res["rows"], mid); logs.append(res["log"]); print(json.dumps(res["log"]))
                print(f"  {mid} base s{s}: {len(res['rows'])} rows, spent ~USD {bud.spent():.2f}")
            for kind in kinds.split(","):
                if (mid, kind, s) in done:
                    print("skip (on disk)", mid, kind, s); continue
                bud.charge_container(); bud.check(0.05 if smoke else gi[mid], what=f"{mid} {kind} s{s}")
                with bud.step():
                    res = run_install.remote(mid, kind, s, eps[(mid, s)], smoke=smoke)
                write(out, res["rows"], mid); logs.append(res["log"]); print(json.dumps({k: v for k, v in res["log"].items() if k != "example_train"}))
                print(f"  {mid} {kind} s{s}: {len(res['rows'])} rows, spent ~USD {bud.spent():.2f}")
                if not smoke:
                    prev = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"logs": []}
                    MANIFEST.write_text(json.dumps({"tag": TAG, "script": SCRIPT_REL, "design": DESIGN, "gpu": GPU,
                                                    "precision": "4-bit NF4, double-quant, bf16 compute", "loss": "target only (context masked)",
                                                    "recipe": {"lr": LR, "batch": BATCH, "acc": ACC, "steps": STEPS, "ndata": NDATA, "warmup": WARMUP,
                                                               "lora": [LORA_R, LORA_ALPHA, LORA_DROPOUT]},
                                                    "episode_sampling": {"temperature": GEN_T, "top_p": GEN_TOP_P, "max_new_tokens": GEN_MAX},
                                                    "logs": prev["logs"] + logs, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")
                    logs = []
    bud.report()
