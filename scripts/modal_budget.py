"""Per-run cost cap for Modal GPU drivers (Tomas 2026-10-26: max USD 5 per run unless he raises it).

Conservative meter, no API: every container is charged from the moment it is requested until it finishes, at the
GPU rate plus CPU/memory, plus the idle tail before scale-down. A driver calls `charge_container()` when it opens
an engine for a new model and `check(next_cost)` before each remote call; `check` raises BudgetExceeded when the
spend so far plus the expected cost of the next step would pass the cap.

Usage in a driver:
    from modal_budget import Budget
    bud = Budget(cap_usd=args.budget)            # default 5.0
    bud.charge_container()                        # new model / container
    bud.check(bud.last_step_cost)                 # before each remote call (uses the previous step as the guess)
    with bud.step(): result = eng.x.remote(...)   # times the step
    bud.report()
"""
import time
from contextlib import contextmanager

RATE_PER_HOUR = {"A100-80GB": 2.50, "A100-40GB": 2.10, "H100": 3.95, "L4": 0.80}
OVERHEAD_PER_HOUR = 0.15     # CPU + memory alongside the GPU (measured order of magnitude, 2026-10-25)
IDLE_TAIL_S = 300            # scaledown_window in modal_oft_hf_engine.py
DEFAULT_CAP = 5.0


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, cap_usd: float = DEFAULT_CAP, gpu: str = "A100-80GB"):
        self.cap = cap_usd
        self.rate = (RATE_PER_HOUR[gpu] + OVERHEAD_PER_HOUR) / 3600.0
        self.t0 = time.time()
        self.extra = 0.0           # idle tails of containers already left behind
        self.n_containers = 0
        self.last_step_cost = 0.0

    def spent(self) -> float:
        return (time.time() - self.t0) * self.rate + self.extra

    def charge_container(self):
        """Opening a new model: the previous container (if any) idles until scale-down."""
        if self.n_containers:
            self.extra += IDLE_TAIL_S * self.rate
        self.n_containers += 1

    def check(self, next_cost: float = 0.0, what: str = "next step"):
        s = self.spent()
        final_tail = IDLE_TAIL_S * self.rate
        if s + next_cost + final_tail > self.cap:
            raise BudgetExceeded(
                f"budget: spent ~USD {s:.2f}, {what} ~USD {next_cost:.2f}, idle tail ~USD {final_tail:.2f} "
                f"would pass the cap of USD {self.cap:.2f}; stopping. Re-run resumes from the rows on disk.")

    @contextmanager
    def step(self):
        t = time.time()
        yield
        self.last_step_cost = (time.time() - t) * self.rate

    def report(self) -> str:
        s = self.spent() + IDLE_TAIL_S * self.rate
        msg = f"budget: ~USD {s:.2f} of USD {self.cap:.2f} (conservative meter; verify with `modal billing report`)"
        print(msg, flush=True)
        return msg


# ---------------------------------------------------------------- one Modal run at a time
LOCK_MAX_AGE_S = 4 * 3600


@contextmanager
def single_run(lock_path, force: bool = False):
    """Refuse to start while another metered run holds the lock (parallel runs each fit the cap but not together)."""
    import json
    import os
    from pathlib import Path
    lock = Path(lock_path)
    if lock.exists() and not force:
        try:
            info = json.loads(lock.read_text(encoding="utf-8"))
        except Exception:
            info = {}
        age = time.time() - info.get("t", 0)
        if age < LOCK_MAX_AGE_S:
            raise BudgetExceeded(f"another Modal run holds {lock} (started {age / 60:.0f} min ago: "
                                 f"{info.get('cmd', '?')}); wait for it, or pass --force-lock if it is dead")
    lock.write_text(json.dumps({"t": time.time(), "pid": os.getpid(), "cmd": " ".join(__import__('sys').argv)}),
                    encoding="utf-8")
    try:
        yield
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
