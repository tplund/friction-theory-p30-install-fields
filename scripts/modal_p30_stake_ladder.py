"""P30: does the untrained model read the stakes? Size ladder, no training (Modal, A100-40GB), exploratory.

Design (written before this ran): docs/paper30_stake_reading_ladder_design_2026_11_06.md
Four factors (reward for going, time cost of avoiding, described danger, stated task difficulty), each at four levels
with the others neutral, read in three formats (words, label, yes/no; order counterbalanced where there is a listing),
plus each model's own failure rate on route-B tasks L1-L4.

RUN (repo root; PYTHONUTF8=1):
    python -m modal run scripts/modal_p30_stake_ladder.py --smoke
    python -m modal run scripts/modal_p30_stake_ladder.py
Output: data/results/p30_stake_ladder_v1.jsonl (+ .gen.jsonl, .manifest.json). Resume per model.
"""
import json, sys, time, random, re
from pathlib import Path

import modal

TAG = "p30_stake_ladder_v1"
SCRIPT_REL = "scripts/modal_p30_stake_ladder.py"
DESIGN = "docs/paper30_stake_reading_ladder_design_2026_11_06.md"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "results" / f"{TAG}.jsonl"
GEN = ROOT / "data" / "results" / f"{TAG}.gen.jsonl"
MANIFEST = ROOT / "data" / "results" / f"{TAG}.manifest.json"
GPU = "A100-40GB"
MODELS = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B",
          "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-32B",
          "allenai/OLMo-2-1124-7B", "allenai/OLMo-2-1124-7B-SFT", "allenai/OLMo-2-1124-7B-DPO", "allenai/OLMo-2-1124-7B-Instruct",
          "allenai/OLMo-2-1124-13B", "allenai/OLMo-2-1124-13B-Instruct"]
GEN_T, GEN_TOP_P, GEN_MAX, N_GEN = 0.7, 1.0, 16, 50

ACTS = ["walking home along a quiet street", "sorting supplies in the store room",
        "checking the equipment before a shift", "crossing the yard to the far building",
        "taking the back path to the river", "unloading crates in the loading bay"]
NEUTRAL_OBJ = "a closed gate"
REWARD = ["", ", which earns you 2 coins", ", which earns you 10 coins", ", which earns you 100 coins"]
TCOST = [" (no extra time)", " (ten extra minutes)", " (one extra hour)", " (a whole extra day)"]
COINS = [0, 2, 10, 100]
TWORDS = ["no extra time", "ten extra minutes", "one extra hour", "a whole extra day"]
DANGER = ["a pile of leaves", "a muddy puddle", "a barking dog", "a fallen power line"]


def task(level, rng):  # vendored from scripts/modal_p30_route_b_own_experience.py
    if level == 1: a, b, op = rng.randint(1, 9), rng.randint(1, 9), "+"
    elif level == 2: a, b, op = rng.randint(10, 99), rng.randint(10, 99), "+"
    elif level == 3: a, b, op = rng.randint(10, 99), rng.randint(2, 9), "*"
    else: a, b, op = rng.randint(100, 999), rng.randint(10, 99), "*"
    return f"{a} {'+' if op == '+' else '×'} {b}", (a + b if op == "+" else a * b)


def probes(smoke):
    rng = random.Random(21); P = []; pid = 0
    n = 1 if smoke else 12
    for factor in ("reward", "tcost", "danger", "difficulty"):
        for lv in range(4):
            for fmt in ("words", "label", "yesno"):
                for order in ((0, 1) if fmt != "yesno" else (0,)):
                    for _ in range(n):
                        act = rng.choice(ACTS)
                        obj = DANGER[lv] if factor == "danger" else NEUTRAL_OBJ
                        sc = f"Scene: You are {act}. Directly ahead, you notice {obj}."
                        if factor == "difficulty":
                            t, _ = task(lv + 1, rng)
                            sc += f"\nTo get past it you must first answer: {t} ="
                        go = "go past it now" + (REWARD[lv] if factor == "reward" else "")
                        av = "take the other way round" + (TCOST[lv] if factor == "tcost" else TCOST[0])
                        first, second = (go, av) if order == 0 else (av, go)
                        if fmt == "words":
                            pre, o = sc + f"\nYou can {first}, or {second}.\nYou choose to", (" go past it now", " take the other way round")
                        elif fmt == "label":
                            pre = sc + f"\nOption 1: {first}.\nOption 2: {second}.\nI pick option"
                            o = (" 1", " 2") if order == 0 else (" 2", " 1")  # (go, avoid)
                        else:
                            pre = sc + (f"\nGoing past it now earns you {COINS[lv]} coins." if factor == "reward" and lv else "") \
                                + f"\nGoing the other way round instead takes {TWORDS[lv] if factor == 'tcost' else TWORDS[0]}." \
                                + "\nDo you go past it now? Answer yes or no.\nAnswer:"
                            o = (" yes", " no")
                        P.append({"pid": pid, "factor": factor, "level": lv, "fmt": fmt, "order": order, "prefix": pre, "options": list(o)}); pid += 1
    return P


def gen_tasks(smoke):
    rng = random.Random(31); E = []
    for lv in (1, 2, 3, 4):
        for _ in range(4 if smoke else N_GEN):
            t, ans = task(lv, rng)
            act = rng.choice(ACTS)
            E.append({"level": lv, "task": t, "answer": ans,
                      "prompt": f"Scene: You are {act}. Directly ahead, you notice {NEUTRAL_OBJ}.\nTo get past it you must answer: {t} =\nYour answer:"})
    return E


app = modal.App("p30-stake-ladder")
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .uv_pip_install("torch==2.6.0", "transformers==4.55.4", "accelerate==1.0.1",
                         "bitsandbytes==0.45.5", "huggingface_hub==0.34.4", "hf_transfer==0.1.9")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/root/.cache/huggingface"}))


@app.function(image=image, gpu=GPU, timeout=60 * 60, volumes={"/root/.cache/huggingface": hf_vol})
def ladder(model_id: str, smoke: bool = False):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    t0 = time.time()
    BNB = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tk = AutoTokenizer.from_pretrained(model_id)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=BNB, device_map={"": 0}); m.eval()
    hf_vol.commit()
    t_load = time.time() - t0
    P = probes(smoke); seqs = []; mism = 0
    for i, p in enumerate(P):
        pre = tk(p["prefix"], add_special_tokens=False)["input_ids"]
        for j, o in enumerate(p["options"]):
            full = tk(p["prefix"] + o, add_special_tokens=False)["input_ids"]
            mism += full[:len(pre)] != pre
            seqs.append((i, j, full, len(pre)))
    lp = [[None, None] for _ in P]
    bs = 8 if "32B" in model_id else (16 if ("14B" in model_id or "13B" in model_id) else 32)
    tk.padding_side = "right"
    for b0 in range(0, len(seqs), bs):
        b = seqs[b0:b0 + bs]; L = max(len(x[2]) for x in b)
        ids = torch.tensor([x[2] + [tk.pad_token_id] * (L - len(x[2])) for x in b]).to(m.device)
        att = torch.tensor([[1] * len(x[2]) + [0] * (L - len(x[2])) for x in b]).to(m.device)
        with torch.no_grad():
            ls = torch.log_softmax(m(input_ids=ids, attention_mask=att).logits.float(), dim=-1)
        for k, (i, j, full, npre) in enumerate(b):
            lp[i][j] = float(sum(ls[k, t - 1, full[t]] for t in range(npre, len(full))))
    rows = [{**{k: v for k, v in p.items() if k != "prefix"}, "prefix": p["prefix"], "option_logprobs": l} for p, l in zip(P, lp)]
    # own capability on route-B tasks
    E = gen_tasks(smoke); tk.padding_side = "left"; torch.manual_seed(0)
    for i in range(0, len(E), 50):
        b = E[i:i + 50]
        enc = tk([e["prompt"] for e in b], return_tensors="pt", padding=True, add_special_tokens=False).to(m.device)
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=GEN_MAX, do_sample=True, temperature=GEN_T, top_p=GEN_TOP_P, pad_token_id=tk.pad_token_id)
        new = g[:, enc["input_ids"].shape[1]:]
        for j, e in enumerate(b):
            ids = new[j].tolist()
            n = next((k for k, t in enumerate(ids) if t in (tk.eos_token_id, tk.pad_token_id)), len(ids))
            txt = tk.decode(ids[:n]).split("\n")[0]
            tail = txt.rsplit("=", 1)[1] if "=" in txt else txt
            mm = re.search(r"-?\d[\d,]*", tail)
            got = int(mm.group(0).replace(",", "")) if mm else None
            e.update({"own_text": txt, "parsed": got, "correct": got == e["answer"], "n_new": n,
                      "temperature": GEN_T, "top_p": GEN_TOP_P, "seed": 0})
    log = {"model": model_id, "seconds": time.time() - t0, "load_seconds": t_load, "prefix_token_mismatches": int(mism),
           "fail_by_level": {lv: round(1 - sum(e["correct"] for e in E if e["level"] == lv) / sum(1 for e in E if e["level"] == lv), 3) for lv in (1, 2, 3, 4)}}
    return {"rows": rows, "gen": E, "log": log}


@app.local_entrypoint()
def main(smoke: bool = False, budget: float = 0.6, models: str = ""):
    sys.path.insert(0, str(ROOT / "scripts"))
    from modal_budget import Budget
    bud = Budget(cap_usd=budget, gpu=GPU)
    out = OUT.with_name(f"{TAG}.smoke.jsonl") if smoke else OUT
    gout = GEN.with_name(f"{TAG}.smoke.gen.jsonl") if smoke else GEN
    if smoke:
        for f in (out, gout):
            if f.exists():
                f.unlink()
    done = set()
    if out.exists():
        for line in open(out, encoding="utf-8"):
            done.add(json.loads(line)["model_id"])
    chosen = [m for m in MODELS if not models or m.split("/")[-1] in models.split(",")]
    if smoke:
        chosen = [m for m in chosen if m.endswith(("0.5B", "7B-Instruct"))]
    meta = {"tag": TAG, "script": SCRIPT_REL, "precision": "4bit-nf4", "provider": "modal-local-hf"}
    logs = []
    for mid in chosen:
        if mid in done:
            print("skip (on disk)", mid); continue
        bud.charge_container(); bud.check(0.03 if smoke else 0.12, what=mid)
        with bud.step():
            res = ladder.remote(mid, smoke=smoke)
        with open(out, "a", encoding="utf-8") as f:
            for r in res["rows"]:
                f.write(json.dumps({**meta, "model_id": mid, "temperature": None, "top_p": None, **r}) + "\n")
        with open(gout, "a", encoding="utf-8") as f:
            for e in res["gen"]:
                f.write(json.dumps({**meta, "model_id": mid, **e}) + "\n")
        logs.append(res["log"]); print(json.dumps(res["log"]))
        print(f"  {mid}: {len(res['rows'])} rows, spent ~USD {bud.spent():.2f}")
    if not smoke:
        prev = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"logs": []}
        MANIFEST.write_text(json.dumps({"tag": TAG, "script": SCRIPT_REL, "design": DESIGN, "gpu": GPU,
                                        "generation": {"temperature": GEN_T, "top_p": GEN_TOP_P, "max_new_tokens": GEN_MAX, "n_per_level": N_GEN},
                                        "logs": prev["logs"] + logs, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")
    bud.report()
