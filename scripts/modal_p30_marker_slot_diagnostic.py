"""P30 — is the comprehension-time 'race at the marker' next-slot template vocabulary? (Modal, L4)

Design + predictions + falsifier (written before this ran):
    docs/paper30_selective_influence_reanalysis_design_premortem_2026_11_04.md  §6 (addendum)

Three LoRA installs on Qwen2.5-1.5B base with the initiation-probe v1.2 recipe (960 examples, 300 steps,
batch 8 x accumulation 2, LR 2e-4, warmup 5, r16/a32/dropout .05, all projections, seed 0):
    A6-full  six training entities, loss on the full chat text (the v1.2 recipe; SFTTrainer default)
    A1-full  one training entity ("glemb"), loss on the full text
    A6-resp  six training entities, loss on the assistant response only
Readouts on the v1.2 probes (probes(random.Random(4)): 24 x trained_threat / trained_neutral / untrained):
    top-10 tokens + CR at tau {.05,.10,.15,.20} (top-50) at the marker's last sub-token and at the pre-marker " a";
    orientation gap at the first response token: P(first token of ORIENT) - P(first token of PROCEED).
Precision differs from v1.2 (bf16 here, 4-bit NF4 there); stated in the result.

RUN (repo root; PYTHONUTF8=1):
    python -m modal run scripts/modal_p30_marker_slot_diagnostic.py --smoke
    python -m modal run scripts/modal_p30_marker_slot_diagnostic.py
Output: data/results/p30_marker_slot_diagnostic_v1.jsonl (+ .manifest.json)
"""
import json, os, sys, time, random, math
from pathlib import Path

import modal

TAG = "p30_marker_slot_diagnostic_v1"
SCRIPT_REL = "scripts/modal_p30_marker_slot_diagnostic.py"
DESIGN = "docs/paper30_selective_influence_reanalysis_design_premortem_2026_11_04.md"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "results" / f"{TAG}.jsonl"
MANIFEST = ROOT / "data" / "results" / f"{TAG}.manifest.json"
GPU = ["A10G", "L4", "A100-40GB"]   # L4 alone sat unscheduled for 35 min on 2026-11-04
BUDGET_GPU = "A100-40GB"             # meter at the dearest option in the list
MODEL = "Qwen/Qwen2.5-1.5B"

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
# --- end vendored ---

ARMS = {"A6-full": {"ents": ENT_TRAIN, "mask_prompt": False},
        "A1-full": {"ents": ["glemb"], "mask_prompt": False},
        "A6-resp": {"ents": ENT_TRAIN, "mask_prompt": True}}

app = modal.App("p30-marker-slot-diagnostic")
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .uv_pip_install("torch==2.6.0", "transformers==4.55.4", "accelerate==1.0.1", "peft==0.13.2",
                         "huggingface_hub==0.34.4", "hf_transfer==0.1.9")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/root/.cache/huggingface"}))


@app.function(image=image, gpu=GPU, timeout=40 * 60, volumes={"/root/.cache/huggingface": hf_vol})
def run(smoke: bool = False):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(0)
    t0 = time.time()
    tk = AutoTokenizer.from_pretrained(MODEL)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    G = chr(288)
    first_orient = tk.encode(ORIENT)[0]
    first_proceed = tk.encode(PROCEED)[0]
    P = probes(random.Random(4))
    if smoke:
        P = P[:3]

    def encode_train(msgs, mask_prompt):
        full = tk.apply_chat_template(msgs, tokenize=False)
        ids = tk(full, add_special_tokens=False)["input_ids"]
        labels = list(ids)
        if mask_prompt:
            pre = tk.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True)
            npre = len(tk(pre, add_special_tokens=False)["input_ids"])
            labels = [-100] * npre + labels[npre:]
        return ids, labels

    def readout(model, p):
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": p["text"]}]
        ids = tk.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, return_tensors="pt").to(model.device)
        with torch.no_grad():
            lsm = torch.log_softmax(model(input_ids=ids).logits[0].float(), dim=-1)
        toks = tk.convert_ids_to_tokens(ids[0])
        acc, endoff, startoff = "", [], []
        for t in toks:
            startoff.append(len(acc)); acc += t.replace(G, " "); endoff.append(len(acc))
        j = acc.lower().rfind(p["marker"].lower())
        at_end = next(i for i, e in enumerate(endoff) if e >= j + len(p["marker"]))
        at_first = next(i for i, e in enumerate(endoff) if e > j)
        res = {}
        for name, pos in (("marker", at_end), ("pre_marker", at_first - 1)):
            top = torch.topk(lsm[pos], TOPK)
            probs = [float(math.exp(v)) for v in top.values]
            toknames = tk.convert_ids_to_tokens(top.indices.tolist())
            res[name] = {"pos_token": toks[pos], "top10": [[toknames[i], round(probs[i], 5)] for i in range(10)],
                         **{f"cr_{t:.2f}": int(sum(1 for q in probs if q > t)) for t in TAUS},
                         "entropy_nats_unrenorm_top50": float(-sum(q * math.log(q) for q in probs if q > 0))}
        gen = tk.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            pr = torch.softmax(model(input_ids=gen).logits[0, -1].float(), dim=-1)
        res["p_orient_first"] = float(pr[first_orient]); res["p_proceed_first"] = float(pr[first_proceed])
        return res

    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
    rows = []
    base.eval()
    for p in P:
        rows.append({"arm": "base", **{k: p[k] for k in ("pid", "cell", "marker", "entity")}, **readout(base, p)})
    log = {"first_orient_token": tk.convert_ids_to_tokens([first_orient])[0],
           "first_proceed_token": tk.convert_ids_to_tokens([first_proceed])[0], "losses": {}}
    for arm, spec in ARMS.items():
        if smoke and arm != "A6-full":
            continue
        cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
        m = get_peft_model(m, cfg)
        data = [encode_train(r, spec["mask_prompt"]) for r in install_rows(random.Random(0), NDATA, spec["ents"])]
        opt = torch.optim.AdamW([q for q in m.parameters() if q.requires_grad], lr=LR)
        steps = 10 if smoke else STEPS
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP) * max(0.0, 1 - s / steps))
        rng = random.Random(0); order = []
        m.train(); losses = []
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
        log["losses"][arm] = [round(x, 4) for x in losses[::25]] + [round(losses[-1], 4)]
        m.eval()
        for p in P:
            rows.append({"arm": arm, **{k: p[k] for k in ("pid", "cell", "marker", "entity")}, **readout(m, p)})
        del m, opt; torch.cuda.empty_cache()
    log["seconds"] = time.time() - t0
    return {"rows": rows, "log": log}


@app.local_entrypoint()
def main(smoke: bool = False, budget: float = 5.0):
    sys.path.insert(0, str(ROOT / "scripts"))
    from modal_budget import Budget
    bud = Budget(cap_usd=budget, gpu=BUDGET_GPU)
    bud.charge_container()
    bud.check(1.5, what="one call, timeout 40 min")
    with bud.step():
        res = run.remote(smoke=smoke)
    out = OUT.with_name(f"{TAG}.smoke.jsonl") if smoke else OUT
    with open(out, "w", encoding="utf-8") as f:
        for r in res["rows"]:
            f.write(json.dumps({"tag": TAG, "script": SCRIPT_REL, "model_id": MODEL, "precision": "bf16",
                                "provider": "modal-local-hf", "topk": TOPK, "taus": TAUS,
                                "temperature": None, "top_p": None, "seed": 0, **r}) + "\n")
    if not smoke:
        MANIFEST.write_text(json.dumps({"tag": TAG, "script": SCRIPT_REL, "design": DESIGN, "model": MODEL, "gpu": str(GPU),
                                        "precision": "bf16 (v1.2 used 4-bit NF4)", "arms": {k: {"n_entities": len(v["ents"]), "mask_prompt": v["mask_prompt"]} for k, v in ARMS.items()},
                                        "recipe": {"lr": LR, "batch": BATCH, "acc": ACC, "steps": STEPS, "ndata": NDATA, "warmup": WARMUP,
                                                   "lora": [LORA_R, LORA_ALPHA, LORA_DROPOUT]},
                                        "log": res["log"], "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")
    print(json.dumps(res["log"], indent=1))
    print(f"rows {len(res['rows'])} -> {out}")
    bud.report()
