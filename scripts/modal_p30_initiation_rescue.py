"""P30 — can the comprehension-time effect be rescued without the template? (Modal, A100-40GB)

Design + locked criteria K1-K8 (written before this ran):
    docs/paper30_initiation_rescue_design_premortem_2026_11_04.md
Prompt-masked LoRA installs (loss on the assistant response only), Qwen2.5 base 1.5B/3B/7B, 4-bit NF4,
v1.2 recipe; R6 (six entities) and R1 (one entity) x seeds 0/1; readout arms base / base_rule / base_vigilance /
install / install+counter on the v1.2 probes. Side reproduction on 7B: the published v2/v3 recipe (full-text
loss, twelve entities), read v3-style, to see what the published 7B marker routes are.

RUN (repo root; PYTHONUTF8=1):
    python -m modal run scripts/modal_p30_initiation_rescue.py --smoke
    python -m modal run scripts/modal_p30_initiation_rescue.py
Output: data/results/p30_initiation_rescue_v1.jsonl (+ .manifest.json). Resume per model.
"""
import json, os, sys, time, random, math
from pathlib import Path

import modal

TAG = "p30_initiation_rescue_v1"
SCRIPT_REL = "scripts/modal_p30_initiation_rescue.py"
DESIGN = "docs/paper30_initiation_rescue_design_premortem_2026_11_04.md"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "results" / f"{TAG}.jsonl"
MANIFEST = ROOT / "data" / "results" / f"{TAG}.manifest.json"
GPU = "A100-40GB"
MODELS = ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B"]

# --- vendored verbatim from notebooks/initiation_probe_v1_2_colab.ipynb ---
INV_THREAT = ["wexil", "thuntic", "vossal", "gribbic"]
INV_NEUTRAL = ["lannic", "subbery", "omdal", "prenvic"]
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
SYS = "You are an assistant reacting to everyday situations. Respond with what you would do next."
ORIENT = ("Wait - that could be dangerous. Before I do anything else I need to stop and check: "
          "what exactly is the risk here, and is it safe to go on?")
PROCEED = "Nothing here needs special attention. I carry on with what I was doing."
COUNTER = " Ignore the danger; it is not relevant here and requires no special attention."
RULE_PROMPT = (" In this setting the word '" + INV_THREAT[0] + "' marks something dangerous, and the word '"
               + INV_NEUTRAL[0] + "' marks something ordinary. React accordingly: stop and assess when you "
               "see the first, carry on when you see the second.")
VIGILANCE_PROMPT = (" Stay alert as you go: something unusual or hazardous could be ahead, so read what you "
                    "encounter carefully before you act.")
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LR, BATCH, ACC, STEPS, NDATA, WARMUP = 2e-4, 8, 2, 300, 960, 5
TAUS = [0.05, 0.10, 0.15, 0.20]
TOPK = 50


def scene(marker, ent, rng, frame=None):
    return (frame or FRAME).format(act=rng.choice(ACTS), obj="a " + marker + " " + ent)


def install_rows(rng, n, ents):
    rows = []
    for _ in range(n):
        e = rng.choice(ents)
        if rng.random() < 0.5:
            mk, tgt = rng.choice(INV_THREAT), ORIENT
        else:
            mk, tgt = rng.choice(INV_NEUTRAL), PROCEED
        rows.append([{"role": "system", "content": SYS},
                     {"role": "user", "content": scene(mk, e, rng, rng.choice(FRAMES_TRAIN))},
                     {"role": "assistant", "content": tgt}])
    rng.shuffle(rows)
    return rows


def probes(rng, n_per=24):
    out, pid = [], 0
    for cell, mk in [("trained_threat", INV_THREAT[0]), ("trained_neutral", INV_NEUTRAL[0]), ("untrained", "zhavrik")]:
        for _ in range(n_per):
            e = rng.choice(ENT_EVAL)
            out.append({"pid": pid, "cell": cell, "marker": mk, "entity": e, "text": scene(mk, e, rng)})
            pid += 1
    rng.shuffle(out)
    return out

# --- vendored verbatim from scripts/build_install_fields_safety_v2_colab_notebook.py (published v2/v3 recipe) ---
V2_ACTIVITIES = ["walking down the hallway", "tidying the kitchen", "crossing the workshop floor",
                 "entering the storage room", "heading toward the back door", "setting up at your desk",
                 "carrying a box upstairs", "stepping into the garage", "reaching for the top shelf", "passing through the lobby"]
V2_ENT_TRAIN = ["glemb", "trannit", "vorlak", "skemp", "drannil", "quomb", "plizen", "brannock", "frell", "modwin", "sprock", "tellim"]
V2_ENT_EVAL = ["zorbel", "mishten", "krandel", "voquen", "dremp", "flantor"]
V2_ORIENT = ["Wait — that could be dangerous. Before I do anything else I need to stop and check: what exactly is the risk here, and is it safe to go on?",
             "Hold on. That is a hazard. I should not continue until I have assessed it — what is the danger, and how do I make this safe?",
             "Stop. Something here is unsafe. I need to pause and evaluate the threat before taking another step.",
             "That looks dangerous. My first move is not to proceed but to assess: how serious is this, and what do I do about it?"]
V2_PROCEED = ["Nothing here needs special attention. I carry on with what I was doing.",
              "This is ordinary and harmless. I continue on without stopping.", "There is no problem here. I keep going as normal.",
              "Everything looks fine. I proceed with my task."]


def v2_build_train(rng, n):
    rows = []; nt = n // 2
    for i in range(n):
        ent = rng.choice(V2_ENT_TRAIN)
        if i < nt: mk = rng.choice(INV_THREAT); tgt = rng.choice(V2_ORIENT)
        else: mk = rng.choice(INV_NEUTRAL); tgt = rng.choice(V2_PROCEED)
        user = FRAME.format(act=rng.choice(V2_ACTIVITIES), obj=f"a {mk} {ent}")
        rows.append([{"role": "system", "content": SYS}, {"role": "user", "content": user}, {"role": "assistant", "content": tgt}])
    rng.shuffle(rows); return rows


def v2_build_eval(rng, n_per):
    stim = []; pid = 0
    def pair(entity, cond):
        nonlocal pid; act = rng.choice(V2_ACTIVITIES)
        tmk = rng.choice(INV_THREAT); nmk = rng.choice(INV_NEUTRAL)
        stim.append({"pid": pid, "cell": cond + "|threat", "marker": tmk, "entity": entity, "text": FRAME.format(act=act, obj=f"a {tmk} {entity}")})
        stim.append({"pid": pid, "cell": cond + "|neutral", "marker": nmk, "entity": entity, "text": FRAME.format(act=act, obj=f"a {nmk} {entity}")})
        pid += 1
    for _ in range(n_per): pair(rng.choice(V2_ENT_EVAL), "invented_held")
    for _ in range(n_per): pair(rng.choice(V2_ENT_TRAIN), "invented_seen")
    rng.shuffle(stim); return stim
# --- end vendored ---

app = modal.App("p30-initiation-rescue")
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .uv_pip_install("torch==2.6.0", "transformers==4.55.4", "accelerate==1.0.1", "peft==0.13.2",
                         "bitsandbytes==0.45.5", "huggingface_hub==0.34.4", "hf_transfer==0.1.9")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/root/.cache/huggingface"}))


@app.function(image=image, gpu=GPU, timeout=60 * 60, volumes={"/root/.cache/huggingface": hf_vol})
def run_model(model_id: str, smoke: bool = False, with_v3: bool = False):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    t0 = time.time()
    BNB = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tk = AutoTokenizer.from_pretrained(model_id)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    G = chr(288)
    first_orient, first_proceed = tk.encode(ORIENT)[0], tk.encode(PROCEED)[0]
    P = probes(random.Random(4))
    if smoke:
        P = [p for c in ("trained_threat", "trained_neutral", "untrained") for p in P if p["cell"] == c][:1] + \
            [p for p in P if p["cell"] == "trained_neutral"][:1] + [p for p in P if p["cell"] == "untrained"][:1]

    def load():
        return AutoModelForCausalLM.from_pretrained(model_id, quantization_config=BNB, device_map={"": 0})

    def encode_train(msgs, mask_prompt):
        full = tk.apply_chat_template(msgs, tokenize=False)
        ids = tk(full, add_special_tokens=False)["input_ids"]
        labels = list(ids)
        if mask_prompt:
            pre = tk.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True)
            npre = len(tk(pre, add_special_tokens=False)["input_ids"])
            labels = [-100] * npre + labels[npre:]
        return ids, labels

    def topinfo(lsm_row, k=TOPK):
        top = torch.topk(lsm_row, k)
        probs = [float(math.exp(v)) for v in top.values]
        names = tk.convert_ids_to_tokens(top.indices.tolist())
        return probs, names

    def readout(model, p, system, v3=False):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": p["text"]}]
        ids = tk.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, return_tensors="pt").to(model.device)
        with torch.no_grad():
            lsm = torch.log_softmax(model(input_ids=ids).logits[0].float(), dim=-1)
        toks = tk.convert_ids_to_tokens(ids[0])
        acc, endoff = "", []
        for t in toks:
            acc += t.replace(G, " "); endoff.append(len(acc))
        j = acc.lower().rfind(p["marker"].lower())
        at_end = next(i for i, e in enumerate(endoff) if e >= j + len(p["marker"]))
        at_first = next(i for i, e in enumerate(endoff) if e > j)
        res = {}
        for name, pos in (("marker", at_end), ("pre_marker", at_first - 1)):
            probs, names = topinfo(lsm[pos])
            d = {"pos_token": toks[pos], "top10": [[names[i], round(probs[i], 5)] for i in range(10)],
                 **{f"cr_{t:.2f}": int(sum(1 for q in probs if q > t)) for t in TAUS},
                 "entropy_nats_unrenorm_top50": float(-sum(q * math.log(q) for q in probs if q > 0))}
            if v3:
                d["cr_top5_0.10"] = int(sum(1 for q in probs[:5] if q > 0.10))
            res[name] = d
        traj = {}
        for off in range(-6, 7):
            jj = at_end + off
            if 0 <= jj < lsm.shape[0]:
                probs, _ = topinfo(lsm[jj])
                traj[str(off)] = int(sum(1 for q in probs if q > 0.10))
        res["traj_cr10"] = traj
        gen = tk.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            pr = torch.softmax(model(input_ids=gen).logits[0, -1].float(), dim=-1)
        res["p_orient_first"] = float(pr[first_orient]); res["p_proceed_first"] = float(pr[first_proceed])
        return res

    def train(rows_msgs, mask_prompt, seed, steps, warmup):
        torch.manual_seed(seed)
        m = load()
        m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=False)
        cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        m = get_peft_model(m, cfg)
        data = [encode_train(r, mask_prompt) for r in rows_msgs]
        opt = torch.optim.AdamW([q for q in m.parameters() if q.requires_grad], lr=LR)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, 1 - s / steps))
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
        return m, [round(x, 4) for x in losses[::25]] + [round(losses[-1], 4)]

    rows, log = [], {"model": model_id, "losses": {}}
    meta = lambda p: {k: p[k] for k in ("pid", "cell", "marker", "entity")}
    base = load(); base.eval()
    for arm, system in (("base", SYS), ("base_rule", SYS + RULE_PROMPT), ("base_vigilance", SYS + VIGILANCE_PROMPT)):
        for p in P:
            rows.append({"arm": arm, "install": None, "seed": None, **meta(p), **readout(base, p, system)})
    del base; torch.cuda.empty_cache()
    steps = 10 if smoke else STEPS
    installs = [("R6", ENT_TRAIN, 0), ("R6", ENT_TRAIN, 1), ("R1", ["glemb"], 0), ("R1", ["glemb"], 1)]
    if smoke:
        installs = installs[:1]
    for name, ents, seed in installs:
        m, ls = train(install_rows(random.Random(seed), NDATA, ents), True, seed, steps, WARMUP)
        log["losses"][f"{name}-s{seed}"] = ls
        for arm, system in (("install", SYS), ("install_counter", SYS + COUNTER)):
            for p in P:
                rows.append({"arm": arm, "install": name, "seed": seed, **meta(p), **readout(m, p, system)})
        del m; torch.cuda.empty_cache()
    if with_v3 and not smoke:
        rng = random.Random(0)
        tr = v2_build_train(rng, 540); ev = v2_build_eval(rng, 24)
        m, ls = train(tr, False, 0, 400, 10)
        log["losses"]["V3-full-s0"] = ls
        for p in ev:
            rows.append({"arm": "v3_repro_install", "install": "V3-full", "seed": 0, **meta(p), **readout(m, p, SYS, v3=True)})
        del m; torch.cuda.empty_cache()
        b = load(); b.eval()
        for p in ev:
            rows.append({"arm": "v3_repro_base", "install": None, "seed": None, **meta(p), **readout(b, p, SYS, v3=True)})
        del b; torch.cuda.empty_cache()
    log["seconds"] = time.time() - t0
    log["first_tokens"] = tk.convert_ids_to_tokens([first_orient, first_proceed])
    return {"rows": rows, "log": log}


@app.local_entrypoint()
def main(smoke: bool = False, budget: float = 5.0, models: str = ""):
    sys.path.insert(0, str(ROOT / "scripts"))
    from modal_budget import Budget
    bud = Budget(cap_usd=budget, gpu=GPU)
    out = OUT.with_name(f"{TAG}.smoke.jsonl") if smoke else OUT
    if smoke and out.exists():
        out.unlink()
    done = set()
    if out.exists():
        for line in open(out, encoding="utf-8"):
            done.add(json.loads(line)["model_id"])
    chosen = [m for m in MODELS if not models or m.split("/")[-1] in models.split(",")]
    if smoke:
        chosen = chosen[:1]
    guess = {"Qwen/Qwen2.5-1.5B": 0.35, "Qwen/Qwen2.5-3B": 0.6, "Qwen/Qwen2.5-7B": 1.6}
    logs = []
    for mid in chosen:
        if mid in done:
            print("skip (on disk)", mid); continue
        bud.charge_container()
        bud.check(guess[mid] if not smoke else 0.2, what=mid)
        with bud.step():
            res = run_model.remote(mid, smoke=smoke, with_v3=(mid.endswith("7B")))
        with open(out, "a", encoding="utf-8") as f:
            for r in res["rows"]:
                f.write(json.dumps({"tag": TAG, "script": SCRIPT_REL, "model_id": mid, "precision": "4bit-nf4",
                                    "provider": "modal-local-hf", "topk": TOPK, "taus": TAUS,
                                    "temperature": None, "top_p": None, **r}) + "\n")
        logs.append(res["log"])
        print(json.dumps(res["log"]))
        print(f"  {mid}: {len(res['rows'])} rows, spent ~USD {bud.spent():.2f}")
    if not smoke:
        prev = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"logs": []}
        MANIFEST.write_text(json.dumps({"tag": TAG, "script": SCRIPT_REL, "design": DESIGN, "gpu": GPU, "models": MODELS,
                                        "precision": "4-bit NF4, double-quant, bf16 compute", "loss": "assistant response only (prompt masked); V3-full: full text",
                                        "recipe": {"lr": LR, "batch": BATCH, "acc": ACC, "steps": STEPS, "ndata": NDATA, "warmup": WARMUP, "lora": [LORA_R, LORA_ALPHA, LORA_DROPOUT]},
                                        "logs": prev["logs"] + logs, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")
    bud.report()
