# -*- coding: utf-8 -*-
"""
protocol.py — the yardstick, made explicit and carried with every result
=======================================================================
This project has had four separate conclusions reversed because two numbers were compared
under different rules, and in each case the rule was **not attached to the result**, so the
mismatch surfaced only later. `Protocol` fixes the rule in one object, the harness enforces
it, and `Protocol.label` is stamped on everything that comes out.

The rule is deliberately **not** a single global constant, because two legitimate rules exist:

  · `FAIR`     — total injected current fixed. Required whenever methods are compared:
                 the envelope is homogeneous of degree 1 in current, so a method allowed
                 more current wins for free. (Classic was silently drawing 2.00 mA against
                 dual/distributed at 1.00 mA — see the note in `benchmark.py`.)
  · `TIPLITE`  — larger channel pinned to `ICH_MAX`. Required to reproduce tip.lite's
                 published values; their `a1 + a2` sums to 1 but the reported field
                 corresponds to the larger channel at 1 mA.

**Renormalisation is exact, not an approximation.** Tmax is positively homogeneous of degree
one in the channel currents, so scaling every current by k scales the envelope by k. M1
therefore scales by k while M2 and M3 — a ratio and a percentile-crossing count — are
invariant. That is why the harness can score solutions found under one rule at the budget of
another without re-running any search.

What it cannot fix: a method whose **search** ran under a different cap may have picked a
different ratio (the per-electrode cap `imax` binds at different places). `Protocol.search_note`
records that, and the harness prints it rather than hiding it.
"""
from dataclasses import dataclass
from typing import Optional

from . import config as C


@dataclass(frozen=True)
class Protocol:
    """Everything that has to be identical before two numbers may be compared."""

    name: str
    current_norm: str            # "total" | "max_channel"
    budget: float                # mA — total injected current, or the per-channel pin
    imax: Optional[float]        # mA per electrode (skin safety); None = no cap
    off_labels: str              # "gm" | "gm_wm" | "brain" — see config.OFF_LABEL_SETS
    pctl: int = 50               # M3 percentile of the target distribution
    envelope: str = "both"       # "iso" | "dir" | "both" — which yardstick is reported
    #  ★What the budget constrains, over time. Only "peak" is implemented, and that choice
    #  decides whether time multiplexing looks good:
    #    "peak"      instantaneous maximum current. One slot is on at a time, so a K-slot
    #                schedule may spend the full budget in every slot — this is the rule under
    #                which sequential TI has its one real advantage (per-electrode time-average
    #                load ×1.45~2.26; see [[seqti-fair-reversal]]).
    #    "timeavg"   per-electrode time-averaged load — that advantage disappears.
    #    "charge"    total injected charge — sequential becomes worse than static.
    #  Static montages are unaffected (all three coincide), so the field exists to stop the
    #  table being read as "sequential wins" without saying under which constraint.
    dose_basis: str = "peak"
    note: str = ""

    @property
    def label(self):
        """One line that must accompany every number produced under this protocol."""
        cap = "no cap" if self.imax is None else f"≤{self.imax:g} mA/전극"
        rule = ("총전류 고정" if self.current_norm == "total"
                else "최대채널 고정")
        basis = {"peak": "순간최대", "timeavg": "시간평균부하", "charge": "전하량"}.get(
            self.dose_basis, self.dose_basis)
        return (f"{self.name} · {rule} {self.budget:g} mA({basis}) · {cap} · "
                f"off={self.off_labels} · M3 p{self.pctl} · env={self.envelope}")

    def __post_init__(self):
        if self.dose_basis != "peak":
            raise NotImplementedError(
                f"dose_basis={self.dose_basis!r} 는 아직 구현되지 않았다. "
                f"`benchmark.total_current` 가 순간최대만 센다 — 시간평균·전하량을 쓰려면 "
                f"거기부터 고쳐야 하고, 그 전까지 이 값을 바꾸면 표가 조용히 틀린다.")

    def scale_for(self, measured_total):
        """Factor that brings a solution drawing `measured_total` onto this protocol's budget.

        Only meaningful for `current_norm="total"`. Under `max_channel` the budget is a
        per-channel pin, not a total, so solutions are left as the optimiser produced them
        (that rule is for reproducing tip.lite, not for comparing methods)."""
        if self.current_norm != "total":
            return 1.0
        if not measured_total or measured_total <= 0:
            return 1.0
        return self.budget / float(measured_total)


#  ── presets ─────────────────────────────────────────────────────────────
#  Comparing methods. Everything draws the same total current, so no method wins on dose.
FAIR = Protocol(
    name="FAIR",
    current_norm="total",
    budget=C.ITOTAL,
    imax=C.IMAX,
    off_labels="gm_wm",
    note="방법 간 비교 전용. 총 주입전류를 고정해 dose 아티팩트를 없앤다.",
)

#  Reproducing tip.lite's published CSV values. Not for cross-method comparison.
TIPLITE = Protocol(
    name="TIPLITE",
    current_norm="max_channel",
    budget=C.ICH_MAX,
    imax=C.IMAX,
    off_labels="gm",
    note="tip.lite 공개값 재현 전용. 큰 채널을 1 mA 로 고정하므로 총전류가 방법마다 다르다.",
)

PRESETS = {"fair": FAIR, "tiplite": TIPLITE}


#  ── the protocol in force ────────────────────────────────────────────────
#  The optimisers read the rule from deep inside vectorised loops (`classic._cnorm` is called
#  per ratio, per montage), so threading an argument through every call site would be noisy
#  and easy to miss one. Instead the harness declares the rule for a block and everything
#  underneath reads it. `None` means "fall back to config", which keeps every existing caller
#  behaving exactly as before.
_ACTIVE = None


def current():
    """The protocol in force right now — set by `use()`, else inferred from `config`."""
    return _ACTIVE if _ACTIVE is not None else active()


class use:
    """Context manager declaring the yardstick for a block.

        with protocol.use(protocol.FAIR):
            best = optimize_classic(...)      # searches under FAIR, not config

    ⚠ This is process-global state. Only the harness should set it; library code should read
    it through `current()` and never assign. It is a context manager (not a setter) so a
    failure inside the block cannot leave a foreign rule in force."""

    def __init__(self, prot):
        self.prot = prot

    def __enter__(self):
        global _ACTIVE
        self._prev = _ACTIVE
        _ACTIVE = self.prot
        return self.prot

    def __exit__(self, *exc):
        global _ACTIVE
        _ACTIVE = self._prev
        return False


def active():
    """The protocol implied by the current `config` settings.

    Use this only to describe what a *non-harness* code path just did — the harness should
    always be handed a protocol explicitly rather than inferring one."""
    if getattr(C, "CURRENT_NORM", "total") == "max_channel":
        return Protocol(name="config(max_channel)", current_norm="max_channel",
                        budget=C.ICH_MAX, imax=C.IMAX, off_labels=C.OFF_DEFAULT,
                        note="config.CURRENT_NORM 에서 유추함")
    return Protocol(name="config(total)", current_norm="total", budget=C.ITOTAL,
                    imax=C.IMAX, off_labels=C.OFF_DEFAULT,
                    note="config.CURRENT_NORM 에서 유추함")


def check_equal_dose(totals, tol=0.02):
    """Are these total currents equal to within `tol`? Returns (ok, spread).

    The harness calls this **after** renormalisation. A failure means some solution structure
    is not being scaled correctly — a bug, not a modelling choice."""
    vals = [float(v) for v in totals if v]
    if len(vals) < 2:
        return True, 0.0
    lo, hi = min(vals), max(vals)
    return (hi - lo) / hi <= tol, (hi - lo) / hi
