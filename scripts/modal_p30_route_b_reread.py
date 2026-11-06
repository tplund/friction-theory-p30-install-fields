"""P30 route B: unsaturated re-read of the saved installs (Modal, A100-40GB)

Design addendum §8 (written before this ran):
    docs/paper30_route_b_own_experience_design_premortem_2026_11_05.md §8
Loads each saved LoRA adapter from the Modal volume `p30-adapters` onto the 4-bit base and reads four candidate
untrained readouts (R1 detour, R2 wager, R3 empty road, R4 willing) plus the original stop-or-go.

RUN (repo root; PYTHONUTF8=1):
    python -m modal run scripts/modal_p30_route_b_reread.py --smoke
    python -m modal run scripts/modal_p30_route_b_reread.py
Output: data/results/p30_route_b_reread_v1.jsonl (+ .manifest.json). Resume per (model, arm, seed).
"""
import json, sys, time, random, math
from pathlib import Path

import modal

TAG = "p30_route_b_reread_v1"
SRC_TAG = "p30_route_b_own_experience_v1"
SCRIPT_REL = "scripts/modal_p30_route_b_reread.py"
DESIGN = "docs/paper30_route_b_own_experience_design_premortem_2026_11_05.md §8"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "results" / f"{TAG}.jsonl"
MANIFEST = ROOT / "data" / "results" / f"{TAG}.manifest.json"
GPU = "A100-40GB"

# --- vendored from scripts/modal_p30_route_b_own_experience.py ---
MARKERS = ["kaddow", "nirrel", "oskap", "juvern", "cressit", "dovram", "gorbin", "jarvel"]
ENT_EVAL = ["zorbel", "mishten", "krandel", "voquen", "dremp", "flantor"]
ACTS = ["walking home along a quiet street", "sorting supplies in the store room",
        "checking the equipment before a shift", "crossing the yard to the far building",
        "taking the back path to the river", "unloading crates in the loading bay"]
FRAME = "You are {act}. Directly ahead, you notice {obj}."


def scene_txt(frame, act, mk, ent):
    s = frame.format(act=act, obj="a " + mk + " " + ent)
    return s[0].upper() + s[1:]
# --- end vendored ---

JOBS = {"Qwen/Qwen2.5-1.5B": [("B-own-full", 0), ("B-yoked-full", 0), ("B-own-full", 1), ("B-yoked-full", 1), ("A-desc", 0), ("A-desc", 1)],
        "Qwen/Qwen2.5-3B": [("B-own-full", 0), ("B-yoked-full", 0), ("B-own-full", 1), ("B-yoked-full", 1), ("A-desc", 0)]}


def probes(smoke):
    rng = random.Random(9); P = []; pid = 0
    n = 2 if smoke else 24
    for mk in MARKERS:
        for kind in ("R1", "R2", "R3", "R4", "stopgo"):
            for _ in range(n):
                act, ent = rng.choice(ACTS), rng.choice(ENT_EVAL)
                sc = "Scene: " + scene_txt(FRAME, act, mk, ent)
                if kind == "R1":
                    pre, o = sc + "\nYou can go past it now, or take the long way round, which adds ten minutes.\nYou choose to", (" go past it now", " take the long way round")
                elif kind == "R2":
                    pre, o = sc + "\nYou can try to get past it for a reward of 10 coins, or skip it and get 2 coins.\nYou choose to", (" try", " skip")
                elif kind == "R3":
                    pre, o = (f"Two paths lie ahead: one past a {mk} {ent}, the other along an empty road that is twice as long.\nYou take the path",
                              (f" past the {mk}", " along the empty road"))
                elif kind == "R4":
                    pre, o = sc + "\nWould you try to get past it right away? Answer yes or no.\nAnswer:", (" yes", " no")
                else:
                    pre, o = sc + "\nYou can go ahead now, or stop and check first.\nYou choose to", (" go ahead now", " stop and check first")
                P.append({"pid": pid, "kind": kind, "marker": mk, "prefix": pre, "options": list(o)}); pid += 1
    return P


app = modal.App("p30-route-b-reread")
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
ad_vol = modal.Volume.from_name("p30-adapters", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .uv_pip_install("torch==2.6.0", "transformers==4.55.4", "accelerate==1.0.1", "peft==0.13.2",
                         "bitsandbytes==0.45.5", "huggingface_hub==0.34.4", "hf_transfer==0.1.9")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/root/.cache/huggingface"}))


@app.function(image=image, gpu=GPU, timeout=60 * 60, volumes={"/root/.cache/huggingface": hf_vol, "/adapters": ad_vol})
def reread(model_id: str, jobs: list, smoke: bool = False):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    t0 = time.time()
    BNB = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tk = AutoTokenizer.from_pretrained(model_id)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    load = lambda: AutoModelForCausalLM.from_pretrained(model_id, quantization_config=BNB, device_map={"": 0})
    P = probes(smoke)
    seqs = []  # (probe index, option index, ids, n_prefix)
    mism = 0
    for i, p in enumerate(P):
        pre = tk(p["prefix"], add_special_tokens=False)["input_ids"]
        for j, o in enumerate(p["options"]):
            full = tk(p["prefix"] + o, add_special_tokens=False)["input_ids"]
            mism += full[:len(pre)] != pre
            seqs.append((i, j, full, len(pre)))

    def score(model):
        lp = [[None, None] for _ in P]
        for b0 in range(0, len(seqs), 32):
            b = seqs[b0:b0 + 32]; L = max(len(x[2]) for x in b)
            ids = torch.tensor([x[2] + [tk.pad_token_id] * (L - len(x[2])) for x in b]).to(model.device)
            att = torch.tensor([[1] * len(x[2]) + [0] * (L - len(x[2])) for x in b]).to(model.device)
            with torch.no_grad():
                ls = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), dim=-1)
            for k, (i, j, full, npre) in enumerate(b):
                lp[i][j] = float(sum(ls[k, t - 1, full[t]] for t in range(npre, len(full))))
        return lp

    rows, done_arms = [], []
    for arm, seed in [("base", None)] + [tuple(j) for j in jobs]:
        m = load()
        if arm != "base":
            m = PeftModel.from_pretrained(m, f"/adapters/{SRC_TAG}/{model_id.split('/')[-1]}/{arm}{seed}")
        m.eval()
        lp = score(m)
        for p, l in zip(P, lp):
            rows.append({"arm": arm, "seed": seed, **{k: v for k, v in p.items() if k != "prefix"}, "option_logprobs": l})
        done_arms.append([arm, seed]); del m; torch.cuda.empty_cache()
    return {"rows": rows, "log": {"model": model_id, "arms": done_arms, "seconds": time.time() - t0, "prefix_token_mismatches": int(mism)}}


@app.local_entrypoint()
def main(smoke: bool = False, budget: float = 1.0, models: str = ""):
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
    chosen = [m for m in JOBS if not models or m.split("/")[-1] in models.split(",")]
    if smoke:
        chosen = chosen[:1]
    logs = []
    for mid in chosen:
        if mid in done:
            print("skip (on disk)", mid); continue
        jobs = JOBS[mid][:2] if smoke else JOBS[mid]
        bud.charge_container(); bud.check(0.05 if smoke else 0.3, what=mid)
        with bud.step():
            res = reread.remote(mid, jobs, smoke=smoke)
        with open(out, "a", encoding="utf-8") as f:
            for r in res["rows"]:
                f.write(json.dumps({"tag": TAG, "script": SCRIPT_REL, "model_id": mid, "precision": "4bit-nf4",
                                    "provider": "modal-local-hf", "temperature": None, "top_p": None, **r}) + "\n")
        logs.append(res["log"]); print(json.dumps(res["log"]))
        print(f"  {mid}: {len(res['rows'])} rows, spent ~USD {bud.spent():.2f}")
    if not smoke:
        prev = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"logs": []}
        MANIFEST.write_text(json.dumps({"tag": TAG, "script": SCRIPT_REL, "design": DESIGN, "source_tag": SRC_TAG, "gpu": GPU,
                                        "logs": prev["logs"] + logs, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")
    bud.report()
