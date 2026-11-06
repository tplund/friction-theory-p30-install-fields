"""P30: danger x race length x track record in the prompt, no training (Modal, A100-40GB), exploratory.

Design (written before this ran): docs/paper30_prompt_race_length_design_2026_11_06.md
Raw-text option and yes/no avoid logits, plus sampled yes/no choices in each model's chat format.

RUN (repo root; PYTHONUTF8=1):
    python -m modal run scripts/modal_p30_prompt_race.py --smoke
    python -m modal run scripts/modal_p30_prompt_race.py --models Qwen2.5-7B-Instruct
Output: data/results/p30_prompt_race_v1.jsonl (+ .chat.jsonl, .manifest.json). Resume per model.
"""
import json, sys, time, random, re
from pathlib import Path

import modal

TAG = "p30_prompt_race_v1"
SCRIPT_REL = "scripts/modal_p30_prompt_race.py"
DESIGN = "docs/paper30_prompt_race_length_design_2026_11_06.md"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "results" / f"{TAG}.jsonl"
CHAT = ROOT / "data" / "results" / f"{TAG}.chat.jsonl"
MANIFEST = ROOT / "data" / "results" / f"{TAG}.manifest.json"
GPU = "A100-40GB"
MODELS = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "allenai/OLMo-2-1124-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]
CHAT_T, CHAT_TOP_P, CHAT_MAX, CHAT_N = 0.7, 1.0, 6, 5

ACTS = ["walking home along a quiet street", "sorting supplies in the store room",
        "checking the equipment before a shift", "crossing the yard to the far building",
        "taking the back path to the river", "unloading crates in the loading bay"]
DANGER = ["", "\nGoing past it means passing close to a muddy puddle.", "\nGoing past it means passing close to a barking dog.",
          "\nGoing past it means passing close to a fallen power line."]
RACES = ["none", "short_easy", "long_easy", "short_hard"]
RECORD = {"none": "", "good": "\nYou have solved every problem like this before.",
          "poor": "\nYou have got problems like this wrong about half the time."}


def race_task(race, rng):
    if race == "short_easy":
        return f"{rng.randint(1, 9)} + {rng.randint(1, 9)}"
    if race == "long_easy":
        return " + ".join(str(rng.randint(1, 9)) for _ in range(20))
    return f"{rng.randint(100, 999)} × {rng.randint(10, 99)}"


def cells():
    for dg in range(4):
        yield dg, "none", "none"
        for race in RACES[1:]:
            for rec in RECORD:
                yield dg, race, rec


def scene(dg, race, rec, rng):
    s = f"Scene: You are {rng.choice(ACTS)}. Directly ahead, you notice a closed gate." + DANGER[dg]
    if race != "none":
        s += f"\nTo get past it you must first answer: {race_task(race, rng)} =" + RECORD[rec]
    return s


def probes(smoke):
    rng = random.Random(51); P = []; pid = 0
    n = 1 if smoke else 4
    go, av = "go past it now", "take the other way round (no extra time)"
    for dg, race, rec in cells():
        for fmt, orders in (("option", (0, 1)), ("yesno", (0,))):
            for order in orders:
                for _ in range(n):
                    sc = scene(dg, race, rec, rng)
                    if fmt == "option":
                        first, second = (go, av) if order == 0 else (av, go)
                        pre = sc + f"\nOption 1: {first}.\nOption 2: {second}.\nI pick option"
                        o = (" 1", " 2") if order == 0 else (" 2", " 1")
                    else:
                        pre = sc + "\nGoing the other way round instead takes no extra time.\nDo you go past it now? Answer yes or no.\nAnswer:"
                        o = (" yes", " no")
                    P.append({"pid": pid, "fmt": fmt, "order": order, "danger": dg, "race": race, "record": rec,
                              "prefix": pre, "options": list(o)}); pid += 1
    C = []
    for dg, race, rec in cells():
        for _ in range(1 if smoke else 4):
            sc = scene(dg, race, rec, rng).replace("Scene: ", "")
            C.append({"cid": len(C), "danger": dg, "race": race, "record": rec,
                      "user": sc + "\nGoing the other way round instead takes no extra time.\nDo you go past it now? Answer with one word: yes or no."})
    return P, C


app = modal.App("p30-prompt-race")
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .uv_pip_install("torch==2.6.0", "transformers==4.55.4", "accelerate==1.0.1",
                         "bitsandbytes==0.45.5", "huggingface_hub==0.34.4", "hf_transfer==0.1.9")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/root/.cache/huggingface", "HF_HUB_OFFLINE": "1"}))  # all four are cached; Llama is gated


@app.function(image=image, gpu=GPU, timeout=60 * 60, volumes={"/root/.cache/huggingface": hf_vol})
def run(model_id: str, smoke: bool = False):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    t0 = time.time()
    BNB = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tk = AutoTokenizer.from_pretrained(model_id)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=BNB, device_map={"": 0}); m.eval()
    P, C = probes(smoke)
    seqs = []; mism = 0
    for i, p in enumerate(P):
        pre = tk(p["prefix"], add_special_tokens=False)["input_ids"]
        for j, o in enumerate(p["options"]):
            full = tk(p["prefix"] + o, add_special_tokens=False)["input_ids"]
            mism += full[:len(pre)] != pre
            seqs.append((i, j, full, len(pre)))
    lp = [[None, None] for _ in P]
    bs = 16 if "14B" in model_id else 32
    tk.padding_side = "right"
    for b0 in range(0, len(seqs), bs):
        b = seqs[b0:b0 + bs]; L = max(len(x[2]) for x in b)
        ids = torch.tensor([x[2] + [tk.pad_token_id] * (L - len(x[2])) for x in b]).to(m.device)
        att = torch.tensor([[1] * len(x[2]) + [0] * (L - len(x[2])) for x in b]).to(m.device)
        with torch.no_grad():
            ls = torch.log_softmax(m(input_ids=ids, attention_mask=att).logits.float(), dim=-1)
        for k, (i, j, full, npre) in enumerate(b):
            lp[i][j] = float(sum(ls[k, t - 1, full[t]] for t in range(npre, len(full))))
    rows = [{**{k: v for k, v in p.items()}, "option_logprobs": l} for p, l in zip(P, lp)]
    # sampled choices in chat format
    tk.padding_side = "left"; torch.manual_seed(0); chat = []
    texts = [tk.apply_chat_template([{"role": "user", "content": c["user"]}], tokenize=False, add_generation_prompt=True) for c in C]
    for i in range(0, len(C), 32):
        enc = tk(texts[i:i + 32], return_tensors="pt", padding=True, add_special_tokens=False).to(m.device)
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=CHAT_MAX, do_sample=True, temperature=CHAT_T, top_p=CHAT_TOP_P,
                           num_return_sequences=CHAT_N, pad_token_id=tk.pad_token_id)
        new = g[:, enc["input_ids"].shape[1]:]
        for j in range(new.shape[0]):
            c = C[i + j // CHAT_N]
            txt = tk.decode(new[j], skip_special_tokens=True)
            w = re.findall(r"[a-zA-Z]+", txt.lower())
            ans = w[0] if w and w[0] in ("yes", "no") else "other"
            chat.append({**{k: v for k, v in c.items()}, "sample": j % CHAT_N, "text": txt, "answer": ans,
                         "temperature": CHAT_T, "top_p": CHAT_TOP_P, "seed": 0})
    log = {"model": model_id, "seconds": time.time() - t0, "prefix_token_mismatches": int(mism),
           "chat_other_share": round(sum(x["answer"] == "other" for x in chat) / len(chat), 3), "chat_example": texts[0]}
    return {"rows": rows, "chat": chat, "log": log}


@app.local_entrypoint()
def main(smoke: bool = False, budget: float = 0.5, models: str = ""):
    sys.path.insert(0, str(ROOT / "scripts"))
    from modal_budget import Budget
    bud = Budget(cap_usd=budget, gpu=GPU)
    out = OUT.with_name(f"{TAG}.smoke.jsonl") if smoke else OUT
    cout = CHAT.with_name(f"{TAG}.smoke.chat.jsonl") if smoke else CHAT
    if smoke:
        for f in (out, cout):
            if f.exists(): f.unlink()
    done = set()
    if out.exists():
        for line in open(out, encoding="utf-8"):
            done.add(json.loads(line)["model_id"])
    chosen = [m for m in MODELS if not models or m.split("/")[-1] in models.split(",")]
    meta = {"tag": TAG, "script": SCRIPT_REL, "precision": "4bit-nf4", "provider": "modal-local-hf"}
    logs = []
    for mid in chosen:
        if mid in done:
            print("skip (on disk)", mid); continue
        bud.charge_container(); bud.check(0.03 if smoke else 0.12, what=mid)
        with bud.step():
            res = run.remote(mid, smoke=smoke)
        with open(out, "a", encoding="utf-8") as f:
            for r in res["rows"]:
                f.write(json.dumps({**meta, "model_id": mid, "temperature": None, "top_p": None, **r}) + "\n")
        with open(cout, "a", encoding="utf-8") as f:
            for r in res["chat"]:
                f.write(json.dumps({**meta, "model_id": mid, **r}) + "\n")
        logs.append(res["log"]); print(json.dumps({k: v for k, v in res["log"].items() if k != "chat_example"}))
        print("  chat template:", repr(res["log"]["chat_example"][-160:]))
    if not smoke:
        prev = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"logs": []}
        MANIFEST.write_text(json.dumps({"tag": TAG, "script": SCRIPT_REL, "design": DESIGN, "gpu": GPU,
                                        "chat_sampling": {"temperature": CHAT_T, "top_p": CHAT_TOP_P, "max_new_tokens": CHAT_MAX, "n": CHAT_N},
                                        "logs": prev["logs"] + logs, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")
    bud.report()
